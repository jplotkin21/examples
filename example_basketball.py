#!/usr/bin/env python

from scipy.stats import binom
import numpy as np
import math
import logging
import argparse


logging.basicConfig(level=logging.INFO)


def parse_args():
    _desc = "determine the likelihood of winning if each team only shoots 2's or 3's resp"
    parser = argparse.ArgumentParser(description=_desc)
    parser.add_argument('shot_attempts', type=int, help='number of shots each team takes')
    parser.add_argument('prob_three', type=float, help='probability of making a three')
    parser.add_argument('prob_two', type=float, help='probability of making a two')
    parsed_args = parser.parse_args()

    return parsed_args


def game_outcome_probabilities(shot_attempts, prob_three, prob_two):
    """probability of a three point only shooting team beating a two point only shooting team
    x_N_pt = number of baskets made for the N point shooting team
    Prob(three_point_team_wins) =
    Prob(two_point_team_points < three_point_team_points) = P(2*x_two_pt < 3*x_three_pt) =
    P(x_two_pt < 1.5*x_three_pt) =
    Sum[n=0..trials]P(x_two_pt < 1.5*x_three_pt|x_three_pt = n)P(x_three_pt = n)

    Args:
        shot_attempts (int): number of shots taken by each team
        prob_three (float): probability of score for the three point taking team
        prob_two (float): probability of score for the two point taking team

    Returns:
        dict: probability that team 3 beats team 2
    """
    prob_three_wins = 0
    prob_tie = 0
    for made_threes in np.arange(shot_attempts + 1):
        prob_made_threes = binom.pmf(made_threes, shot_attempts, prob_three)
        max_two_made_still_lose = math.floor(1.5*made_threes)
        if 2 * max_two_made_still_lose == 3 * made_threes:
            prob_tie = (prob_tie
                        + binom.pmf(max_two_made_still_lose, shot_attempts, prob_two)
                        * prob_made_threes)
            max_two_made_still_lose = max_two_made_still_lose - 1
        if max_two_made_still_lose < 0:
            continue
        two_make_cdf = binom.cdf(max_two_made_still_lose, shot_attempts, prob_two)
        prob_three_wins = prob_three_wins + two_make_cdf * prob_made_threes

    return {'three wins': prob_three_wins, 'tie': prob_tie, 'two wins': 1 - prob_three_wins - prob_tie}


if __name__ == '__main__':
    args = parse_args()
    x = game_outcome_probabilities(args.shot_attempts, args.prob_three, args.prob_two)
    logging.info(', '.join(['{}: {}'.format(k, round(v, 4)) for k, v in x.items()]))

