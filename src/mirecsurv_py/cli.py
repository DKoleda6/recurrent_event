"""Simple CLI to run example fits from the command line."""
import argparse
import pandas as pd
from .core import fit_rec_ev_model


def main():
    parser = argparse.ArgumentParser(description='Fit recurrent event model on CSV input')
    parser.add_argument('--input', required=True, help='CSV file with events (one row per event)')
    parser.add_argument('--id', default='id')
    parser.add_argument('--time', default='time')
    parser.add_argument('--event', default='event')
    parser.add_argument('--episode', default='episode')
    parser.add_argument('--covariates', default='x', help='comma-separated covariates')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    covs = args.covariates.split(',') if args.covariates else []
    out = fit_rec_ev_model([df], covariates=covs, id_col=args.id, time_col=args.time,
                            event_col=args.event, episode_col=args.episode, gap_time=True)
    print(out['pooled']['pooled_table'])

if __name__ == '__main__':
    main()