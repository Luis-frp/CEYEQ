import sys
import yaml
from utils.config import apply_interactive_menu, parse_args, run_training

def main():
    args = parse_args()
    if args.interactive:
        args = apply_interactive_menu(args)
    run_training(args)

if __name__ == '__main__':
    main()
