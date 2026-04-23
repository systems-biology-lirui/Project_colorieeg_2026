from common import build_common_parser, execute_decoding_scheme, normalize_common_args, stack_centered_samples


SCHEME_NAME = 'within_pair_centered'
SCHEME_TITLE = 'Task1 within-pair centered color-vs-gray decoding'
SCHEME_NOTE = (
    'Subtract the pair mean from each color and gray trial before decoding. '
    'This removes image-shared content and keeps only the residual domain effect inside each matched pair.'
)


def main():
    parser = build_common_parser('Binary color-vs-gray decoding after within-pair centering.')
    config = normalize_common_args(parser.parse_args())
    summary = execute_decoding_scheme(config, SCHEME_NAME, SCHEME_TITLE, SCHEME_NOTE, stack_centered_samples)
    print(summary)


if __name__ == '__main__':
    main()