from iml1515_broad_sampling import create_parser, run_generation


def parse_args():
    return create_parser("d").parse_args()


def main():
    run_generation(parse_args(), "d")


if __name__ == "__main__":
    main()
