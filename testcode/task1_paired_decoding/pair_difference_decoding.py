from common import build_common_parser, execute_decoding_scheme, normalize_common_args, stack_difference_samples


SCHEME_NAME = 'pair_difference'
SCHEME_TITLE = 'Task1 pair-difference sign decoding'
SCHEME_NOTE = (
    'Build difference vectors from each matched pair using +(color-gray) and -(color-gray). '
    'This is the strongest isolation of the color effect because shared image identity is canceled directly.'
)


def main():
    parser = build_common_parser('Binary color-vs-gray decoding on signed pair-difference features.')
    config = normalize_common_args(parser.parse_args())
    summary = execute_decoding_scheme(config, SCHEME_NAME, SCHEME_TITLE, SCHEME_NOTE, stack_difference_samples)
    print(summary)


if __name__ == '__main__':
    main()