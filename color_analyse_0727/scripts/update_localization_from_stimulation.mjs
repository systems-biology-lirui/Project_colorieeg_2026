import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const ANNOTATION_PATH = path.join(
  PROJECT_ROOT,
  "color_analyse_0727",
  "metadata",
  "stimulation_behavioral_annotation.csv",
);
const SOURCE_ROOT = path.join(PROJECT_ROOT, "processed_data");
const OUTPUT_ROOT = process.env.COLOR_LOCALIZATION_OUTPUT_ROOT || SOURCE_ROOT;

function normalizeLabel(value) {
  return String(value ?? "").trim().toUpperCase().replace(/\s+/g, "");
}

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function mergeRecord(map, row) {
  const contacts = String(row.stimulated_contacts || "")
    .split(";")
    .map(normalizeLabel)
    .filter(Boolean);
  for (const contact of contacts) {
    const current = map.get(contact) || {
      recorded: true,
      pairs: [],
      classes: [],
      summaries: [],
      colorWithSti: false,
      colorReview: false,
    };
    current.pairs.push(row.stim_pair_canonical);
    current.classes.push(row.response_class);
    current.summaries.push(row.behavior_summary);
    current.colorWithSti ||= row.pair_color_with_sti.toLowerCase() === "yes";
    current.colorReview ||= row.pair_color_review.toLowerCase() === "yes";
    map.set(contact, current);
  }
}

async function updateSubject(subject, records) {
  const inputPath = path.join(SOURCE_ROOT, subject, `${subject}_ieegloc.xlsx`);
  const outputPath = path.join(OUTPUT_ROOT, subject, `${subject}_ieegloc.xlsx`);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const sheet = workbook.worksheets.getItemAt(0);
  const used = sheet.getUsedRange();
  const matrix = used.values.map((row) => [...row]);
  const headers = matrix[0].map((value) => String(value ?? "").trim());
  const channelIndex = headers.indexOf("Channel");
  if (channelIndex < 0) throw new Error(`${inputPath} has no Channel column`);

  const targetHeaders = [
    "color_with_sti",
    "stim_behavior_recorded",
    "stim_color_evidence",
    "stim_behavior_pairs",
    "stim_behavior_summary",
    "stim_record_source",
  ];
  const columnIndexes = {};
  for (const header of targetHeaders) {
    let index = headers.indexOf(header);
    if (index < 0) {
      index = headers.length;
      headers.push(header);
      for (const row of matrix) row.push(null);
    }
    columnIndexes[header] = index;
  }

  const byContact = new Map();
  for (const row of records) mergeRecord(byContact, row);
  const source = "visual_experiment/电刺激行为学记录/seeg电刺激行为学记录.xlsx";

  matrix[0] = headers;
  for (let rowIndex = 1; rowIndex < matrix.length; rowIndex += 1) {
    const contact = normalizeLabel(matrix[rowIndex][channelIndex]);
    const record = byContact.get(contact);
    const evidence = record
      ? [...new Set(record.classes)].sort().join(";")
      : "not_in_record";
    matrix[rowIndex][columnIndexes.color_with_sti] = Boolean(record?.colorWithSti);
    matrix[rowIndex][columnIndexes.stim_behavior_recorded] = Boolean(record);
    matrix[rowIndex][columnIndexes.stim_color_evidence] = evidence;
    matrix[rowIndex][columnIndexes.stim_behavior_pairs] = record
      ? [...new Set(record.pairs)].sort().join(";")
      : "";
    matrix[rowIndex][columnIndexes.stim_behavior_summary] = record
      ? [...new Set(record.summaries)].join(" | ")
      : "未出现在电刺激行为记录中";
    matrix[rowIndex][columnIndexes.stim_record_source] = record ? source : "";
  }

  const targetRange = sheet.getRangeByIndexes(0, 0, matrix.length, headers.length);
  targetRange.values = matrix;
  const headerRange = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  headerRange.format.font = { bold: true };
  headerRange.format.horizontalAlignment = "center";
  for (const header of targetHeaders.slice(1)) {
    const index = columnIndexes[header];
    const range = sheet.getRangeByIndexes(0, index, matrix.length, 1);
    range.format.wrapText = true;
    range.format.columnWidth = header === "stim_behavior_summary" ? 32 : 20;
  }
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  return { subject, rows: matrix.length - 1, columns: headers.length, outputPath };
}

const annotation = parseCsv(await fs.readFile(ANNOTATION_PATH, "utf8"));
const subjects = [...new Set(annotation.map((row) => row.subject))].sort();
const summaries = [];
for (const subject of subjects) {
  summaries.push(await updateSubject(subject, annotation.filter((row) => row.subject === subject)));
}

const contactHeaders = [
  "subject",
  "channel",
  "stim_behavior_recorded",
  "color_with_sti",
  "color_review",
  "stim_color_evidence",
  "stim_behavior_pairs",
  "stim_behavior_summary",
  "stim_record_source",
];
const contactRows = [];
for (const subject of subjects) {
  const byContact = new Map();
  for (const row of annotation.filter((item) => item.subject === subject)) mergeRecord(byContact, row);
  for (const [channel, record] of [...byContact.entries()].sort()) {
    contactRows.push({
      subject,
      channel,
      stim_behavior_recorded: true,
      color_with_sti: record.colorWithSti,
      color_review: record.colorReview,
      stim_color_evidence: [...new Set(record.classes)].sort().join(";"),
      stim_behavior_pairs: [...new Set(record.pairs)].sort().join(";"),
      stim_behavior_summary: [...new Set(record.summaries)].join(" | "),
      stim_record_source: "visual_experiment/电刺激行为学记录/seeg电刺激行为学记录.xlsx",
    });
  }
}
const contactPath = path.join(PROJECT_ROOT, "color_analyse_0727", "metadata", "stimulation_behavioral_electrodes.csv");
await fs.writeFile(
  contactPath,
  `${contactHeaders.join(",")}\n${contactRows.map((row) => contactHeaders.map((header) => csvEscape(row[header])).join(",")).join("\n")}\n`,
  "utf8",
);
console.log(JSON.stringify({ outputRoot: OUTPUT_ROOT, subjects: summaries, contactPath }, null, 2));
