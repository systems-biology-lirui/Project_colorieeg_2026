from common import build_common_parser, execute_decoding_scheme, normalize_common_args, stack_raw_samples


SCHEME_NAME = 'paired_group_cv'
SCHEME_TITLE = 'Task1 paired Group-CV color-vs-gray decoding'
SCHEME_NOTE = (
    'Use the original color and gray trials, but keep the two samples from the same image pair '
    'in the same cross-validation fold. This tests whether color-vs-gray generalizes to unseen images.'
)


def main():
    parser = build_common_parser('Binary color-vs-gray decoding with pair-aware grouped CV.')
    config = normalize_common_args(parser.parse_args())
    summary = execute_decoding_scheme(config, SCHEME_NAME, SCHEME_TITLE, SCHEME_NOTE, stack_raw_samples)
    print(summary)


if __name__ == '__main__':
    main()