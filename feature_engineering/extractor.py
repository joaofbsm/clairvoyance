#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Feature extractor and model builder"""

from __future__ import print_function

import os
import sys
import MySQLdb
import pandas as pd
import numpy as np
from tqdm import tqdm

#==================================FUNCTIONS==================================#

def onehot_champions(match, db):
    champions = pd.read_sql("SELECT id FROM Champion", db)
    champions["pos"] = champions.index
    champions = champions.set_index("id").to_dict()

    blue_team = match[:5]
    red_team = match[5:10]

    blue_champions = np.zeros(len(champions["pos"]))
    red_champions = np.zeros(len(champions["pos"]))

    for _, player in blue_team.iterrows():
        blue_champions[champions["pos"][player["championId"]]] = 1

    for _, player in red_team.iterrows():
        red_champions[champions["pos"][player["championId"]]] = 1

    result = np.concatenate((blue_champions, red_champions))
    return result


def onehot_spells(match, db):
    spells = pd.read_sql("SELECT id FROM SummonerSpell "
                         "WHERE name='Barrier' OR name='Cleanse' "
                         "OR name='Exhaust' OR name='Flash' OR name='Ghost' "
                         "OR name='Heal' OR name='Ignite' OR name='Smite' "
                         "OR name='Teleport'", db)
    spells["pos"] = spells.index
    spells = spells.set_index("id").to_dict()

    blue_team = match[:5]
    red_team = match[5:10]

    blue_spells = np.zeros(len(spells["pos"]))
    red_spells = np.zeros(len(spells["pos"]))

    for _, player in blue_team.iterrows():
        blue_spells[spells["pos"][player["spell1Id"]]] += 1
        blue_spells[spells["pos"][player["spell2Id"]]] += 1

    for _, player in red_team.iterrows():
        red_spells[spells["pos"][player["spell1Id"]]] += 1
        red_spells[spells["pos"][player["spell2Id"]]] += 1

    result = np.concatenate((blue_spells, red_spells))
    return result


def onehot_summoner_masteries_team(match, db, cursor):
    masteries = pd.read_sql("SELECT id FROM Mastery", db)
    masteries["pos"] = masteries.index
    masteries = masteries.set_index("id").to_dict()

    get_summoner_masteries = ("SELECT M.masteryId, M.rank "
                              "FROM MatchParticipant P, MatchMastery M, "
                              "MatchDetail D, MatchPlayer PL "
                              "WHERE PL.summonerId = %s "
                              "AND P._match_id = %s "
                              "AND PL._participant_id = P._id "
                              "AND P._id = M._participant_id "
                              "AND P._match_id = D.matchId AND D.mapId = 11 "
                              "ORDER BY P._match_id, PL.summonerId")

    blue_team = match[:5]
    red_team = match[5:10]
    
    blue_summoner_masteries = np.zeros(45)
    red_summoner_masteries = np.zeros(45)

    for _, player in blue_team.iterrows():
        cursor.execute(get_summoner_masteries, (player["summonerId"],
                                                player["matchId"]))
        summoner_masteries = list(cursor)
        for mastery, rank in summoner_masteries:
            blue_summoner_masteries[masteries["pos"][mastery]] += rank

    for _, player in red_team.iterrows():
        cursor.execute(get_summoner_masteries, (player["summonerId"],
                                                player["matchId"]))
        summoner_masteries = list(cursor)
        for mastery, rank in summoner_masteries:
            red_summoner_masteries[masteries["pos"][mastery]] += rank

    results = np.concatenate((blue_summoner_masteries, red_summoner_masteries))
    return results


def dmg_types_team(match, db):
    champion_dmg = pd.read_sql("SELECT _champion_id, attack, defense, magic "
                         "FROM ChampionInfo "
                         "ORDER BY _champion_id", db)
    champion_dmg = champion_dmg.set_index("_champion_id").T.to_dict("list")

    blue_team = match[:5]
    red_team = match[5:10]

    blueteam_dmg = np.zeros((3))
    redteam_dmg = np.zeros((3))


    for _, player in blue_team.iterrows():
        blueteam_dmg += champion_dmg[player["championId"]]

    for _, player in red_team.iterrows():
        redteam_dmg += champion_dmg[player["championId"]]

    result = np.concatenate((blueteam_dmg, redteam_dmg))
    return result


def dmg_types_percent_team(match, db):
    champion_dmg = pd.read_sql("SELECT _champion_id, attack, magic "
                         "FROM ChampionInfo "
                         "ORDER BY _champion_id", db)
    champion_dmg = champion_dmg.set_index("_champion_id").T.to_dict("list")

    blue_team = match[:5]
    red_team = match[5:10]

    blueteam_dmg = np.zeros((2))
    redteam_dmg = np.zeros((2))

    for _, player in blue_team.iterrows():
        blueteam_dmg += champion_dmg[player["championId"]]

    total_dmg = np.sum(blueteam_dmg)
    blueteam_dmg = 100 * np.around(np.divide(blueteam_dmg, total_dmg), 
                                   decimals=5)

    for _, player in red_team.iterrows():
        redteam_dmg += champion_dmg[player["championId"]]

    total_dmg = np.sum(redteam_dmg)
    redteam_dmg = 100 * np.around(np.divide(redteam_dmg, total_dmg), 
                                   decimals=5)

    result = np.concatenate((blueteam_dmg, redteam_dmg))
    return result


def mastery_scores_team(match, cursor):
    get_mastery_scores = ("SELECT mastery "
                          "FROM SummonerMasteries "
                          "WHERE summId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    mastery_scores = np.zeros(2)
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)
        if not mastery_score:
            return None
        mastery_score = mastery_score[0][0]
        mastery_scores[0] += mastery_score


    for _, player in red_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)
        if not mastery_score:
            return None
        mastery_score = mastery_score[0][0]
        mastery_scores[1] += mastery_score

    return mastery_scores


def champion_masteries_team(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    champion_masteries = np.zeros(2)
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)
        if not champion_mastery:
            return None
        champion_mastery = champion_mastery[0][0]
        champion_masteries[0] += champion_mastery


    for _, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)
        if not champion_mastery:
            return None
        champion_mastery = champion_mastery[0][0]
        champion_masteries[1] += champion_mastery

    return champion_masteries


def champion_masteries_summoner(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_champion_masteries = np.zeros(5)
    red_champion_masteries = np.zeros(5)
    
    i = 0
    for _, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)
        if not champion_mastery:
            return None
        champion_mastery = champion_mastery[0][0]
        blue_champion_masteries[i] = champion_mastery
        i += 1

    i = 0
    for _, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)
        if not champion_mastery:
            return None
        champion_mastery = champion_mastery[0][0]
        red_champion_masteries[i] = champion_mastery
        i += 1

    champion_masteries = np.concatenate((blue_champion_masteries, 
                                         red_champion_masteries))

    return champion_masteries


def summoner_wins_and_rate_team(match, cursor):
    get_history = ("SELECT wins, losses "
                    "FROM SummonerHistory "
                    "WHERE summId = %s")

    blue_team = match[:5]
    red_team = match[5:10]

    blue_total = np.zeros(1)
    red_total = np.zeros(1)
    blue_wins = np.zeros(1)
    red_wins = np.zeros(1)
    blue_rate = np.zeros(1)
    red_rate = np.zeros(1)

    for _, player in blue_team.iterrows():
        cursor.execute(get_history, [player["summonerId"]])
        outcomes = list(cursor)[0]
        if not outcomes:
            continue
        wins = outcomes[0]
        losses = outcomes[1]
        blue_total += wins + losses
        blue_wins += wins

    # Harmonic mean
    if blue_total > 0:
        blue_rate = (blue_wins / (blue_total * 1.0)) * 100

    for _, player in red_team.iterrows():
        cursor.execute(get_history, [player["summonerId"]])
        outcomes = list(cursor)[0]
        if not outcomes:
            continue
        wins = outcomes[0]
        losses = outcomes[1]
        red_total += wins + losses
        red_wins += wins

    if red_total > 0:
        red_rate = (red_wins / (red_total * 1.0)) * 100

    result = np.concatenate((blue_rate, blue_wins, red_rate, red_wins))
    return result
    

def champion_wins_and_rate_team(match, cursor):
    get_history = ("SELECT wins, losses "
                    "FROM SummonerChampHistory "
                    "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]

    blue_total = np.zeros(1)
    red_total = np.zeros(1)
    blue_wins = np.zeros(1)
    red_wins = np.zeros(1)
    blue_rate = np.zeros(1)
    red_rate = np.zeros(1)


    for _, player in blue_team.iterrows():
        cursor.execute(get_history, (player["summonerId"], player["championId"]))
        outcomes = list(cursor)[0]
        if not outcomes:
            continue
        wins = outcomes[0]
        losses = outcomes[1]
        blue_total += wins + losses
        blue_wins += wins

    if blue_total > 0:
        blue_rate = (blue_wins / (blue_total * 1.0)) * 100

    for _, player in red_team.iterrows():
        cursor.execute(get_history, (player["summonerId"], player["championId"]))
        outcomes = list(cursor)[0]
        if not outcomes:
            continue
        wins = outcomes[0]
        losses = outcomes[1]
        red_total += wins + losses
        red_wins += wins

    if red_total > 0:
        red_rate = (red_wins / (red_total * 1.0)) * 100

    result = np.concatenate((blue_rate, blue_wins, red_rate, red_wins))
    return result


def team_features_zero_to_ten(match, cursor):
    get_features = ("SELECT PL.summonerId, PTD._type, PTD.zeroToTen "
                    "FROM MatchParticipant PA, MatchPlayer PL, "
                    "MatchParticipantTimeline PT, "
                    "MatchParticipantTimelineData PTD "
                    "WHERE PL.summonerId = %s AND PA._match_id = %s "
                    "AND PL._participant_id = PA._id "
                    "AND PA._id = PT._participant_id "
                    "AND PT._id = PTD._timeline_id")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_zero_to_ten = np.zeros(4)
    red_zero_to_ten = np.zeros(4)

    for _, player in blue_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                blue_zero_to_ten[0] += features[2]
            elif features[1] == "damageTakenPerMinDeltas":
                blue_zero_to_ten[1] += features[2]
            elif features[1] == "goldPerMinDeltas":
                blue_zero_to_ten[2] += features[2]
            elif features[1] == "xpPerMinDeltas":
                blue_zero_to_ten[3] += features[2]

    for _, player in red_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                red_zero_to_ten[0] += features[2]
            elif features[1] == "damageTakenPerMinDeltas":
                red_zero_to_ten[1] += features[2]
            elif features[1] == "goldPerMinDeltas":
                red_zero_to_ten[2] += features[2]
            elif features[1] == "xpPerMinDeltas":
                red_zero_to_ten[3] += features[2]

    zero_to_ten = np.concatenate((blue_zero_to_ten, red_zero_to_ten))

    return zero_to_ten


def team_features_zero_to_twenty(match, cursor):
    get_features = ("SELECT PL.summonerId, PTD._type, PTD.zeroToTen, "
                    "PTD.tenToTwenty "
                    "FROM MatchParticipant PA, MatchPlayer PL, "
                    "MatchParticipantTimeline PT, "
                    "MatchParticipantTimelineData PTD "
                    "WHERE PL.summonerId = %s AND PA._match_id = %s "
                    "AND PL._participant_id = PA._id "
                    "AND PA._id = PT._participant_id "
                    "AND PT._id = PTD._timeline_id")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_zero_to_twenty = np.zeros(4)
    red_zero_to_twenty = np.zeros(4)

    for _, player in blue_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                blue_zero_to_twenty[0] += features[2] + features[3]
            elif features[1] == "damageTakenPerMinDeltas":
                blue_zero_to_twenty[1] += features[2] + features[3]
            elif features[1] == "goldPerMinDeltas":
                blue_zero_to_twenty[2] += features[2] + features[3]
            elif features[1] == "xpPerMinDeltas":
                blue_zero_to_twenty[3] += features[2] + features[3]

    for _, player in red_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                red_zero_to_twenty[0] += features[2] + features[3]
            elif features[1] == "damageTakenPerMinDeltas":
                red_zero_to_twenty[1] += features[2] + features[3]
            elif features[1] == "goldPerMinDeltas":
                red_zero_to_twenty[2] += features[2] + features[3]
            elif features[1] == "xpPerMinDeltas":
                red_zero_to_twenty[3] += features[2] + features[3]

    zero_to_twenty = np.concatenate((blue_zero_to_twenty, red_zero_to_twenty))

    return zero_to_twenty


def team_features_zero_to_thirty(match, cursor):
    get_features = ("SELECT PL.summonerId, PTD._type, PTD.zeroToTen, "
                    "PTD.tenToTwenty, PTD.twentyToThirty "
                    "FROM MatchParticipant PA, MatchPlayer PL, "
                    "MatchParticipantTimeline PT, "
                    "MatchParticipantTimelineData PTD "
                    "WHERE PL.summonerId = %s AND PA._match_id = %s "
                    "AND PL._participant_id = PA._id "
                    "AND PA._id = PT._participant_id "
                    "AND PT._id = PTD._timeline_id")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_zero_to_thirty = np.zeros(4)
    red_zero_to_thirty = np.zeros(4)

    for _, player in blue_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                blue_zero_to_thirty[0] += (features[2] + features[3]
                                           + features[4])
            elif features[1] == "damageTakenPerMinDeltas":
                blue_zero_to_thirty[1] += (features[2] + features[3]
                                           + features[4])
            elif features[1] == "goldPerMinDeltas":
                blue_zero_to_thirty[2] += (features[2] + features[3]
                                           + features[4])
            elif features[1] == "xpPerMinDeltas":
                blue_zero_to_thirty[3] += (features[2] + features[3]
                                           + features[4])

    for _, player in red_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                red_zero_to_thirty[0] += (features[2] + features[3]
                                          + features[4])
            elif features[1] == "damageTakenPerMinDeltas":
                red_zero_to_thirty[1] += (features[2] + features[3]
                                          + features[4])
            elif features[1] == "goldPerMinDeltas":
                red_zero_to_thirty[2] += (features[2] + features[3]
                                          + features[4])
            elif features[1] == "xpPerMinDeltas":
                red_zero_to_thirty[3] += (features[2] + features[3]
                                          + features[4])

    zero_to_thirty = np.concatenate((blue_zero_to_thirty, red_zero_to_thirty))

    return zero_to_thirty


def team_features_zero_to_end(match, cursor):
    get_features = ("SELECT PL.summonerId, PTD._type, PTD.zeroToTen, "
                    "PTD.tenToTwenty, PTD.twentyToThirty, PTD.thirtyToEnd "
                    "FROM MatchParticipant PA, MatchPlayer PL, "
                    "MatchParticipantTimeline PT, "
                    "MatchParticipantTimelineData PTD "
                    "WHERE PL.summonerId = %s AND PA._match_id = %s "
                    "AND PL._participant_id = PA._id "
                    "AND PA._id = PT._participant_id "
                    "AND PT._id = PTD._timeline_id")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_zero_to_end = np.zeros(4)
    red_zero_to_end = np.zeros(4)

    for _, player in blue_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                blue_zero_to_end[0] += (features[2] + features[3]
                                        + features[4] + features[5])
            elif features[1] == "damageTakenPerMinDeltas":
                blue_zero_to_end[1] += (features[2] + features[3]
                                        + features[4] + features[5])
            elif features[1] == "goldPerMinDeltas":
                blue_zero_to_end[2] += (features[2] + features[3]
                                        + features[4] + features[5])
            elif features[1] == "xpPerMinDeltas":
                blue_zero_to_end[3] += (features[2] + features[3]
                                        + features[4] + features[5])

    for _, player in red_team.iterrows():
        cursor.execute(get_features, (player["summonerId"],
                                      player["matchId"]))
        player_features = list(cursor)
        if not player_features:
            return None
        for features in player_features:
            if features[1] == "creepsPerMinDeltas":
                red_zero_to_end[0] += (features[2] + features[3]
                                       + features[4] + features[5])
            elif features[1] == "damageTakenPerMinDeltas":
                red_zero_to_end[1] += (features[2] + features[3]
                                       + features[4] + features[5])
            elif features[1] == "goldPerMinDeltas":
                red_zero_to_end[2] += (features[2] + features[3]
                                       + features[4] + features[5])
            elif features[1] == "xpPerMinDeltas":
                red_zero_to_end[3] += (features[2] + features[3]
                                       + features[4] + features[5])

    zero_to_end = np.concatenate((blue_zero_to_end, red_zero_to_end))

    return zero_to_end


def remove_incomplete_instances(dataset):
    incomplete_instances = []
    for i, instance in enumerate(dataset):
        complete = np.count_nonzero(instance)
        if not complete:
            incomplete_instances.append(i)

    dataset = np.delete(dataset, incomplete_instances, axis=0)
    print("\n\n", len(incomplete_instances), "incomplete instances removed.")

    return dataset

#===================================MODELS====================================#

def feature_testing(db, cursor):
    """
    Size of features
    ----------------

    onehot_champions: 272
    onehot_spells: 18
    onehot_summoner_masteries_team: 90
    dmg_types_team = 6
    dmg_types_percent_team = 4
    mastery_scores_team = 2
    mastery_scores_diff = 1
    champion_masteries_team = 2
    champion_masteries_summoner = 10
    summoner_wins_and_rate_team = 4
    champion_wins_and_rate_team = 4
    zero_to_ten -> end = 8
    zero_to_ten -> end_diff = 4
    winner: 1

    TOTAL: 418
    """

    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " P.spell1Id, P.spell2Id, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId", db)

    dataset = np.zeros((df.shape[0] / 10, 407))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        #=================================PRE=================================#
        champions = onehot_champions(match, db)
        spells = onehot_spells(match, db)
        masteries = onehot_summoner_masteries_team(match, db, cursor)
        dmg_types = dmg_types_team(match, db)
        dmg_percent = dmg_types_percent_team(match, db)
        mastery_scores = mastery_scores_team(match, cursor)
        if mastery_scores is None:
            continue
        mastery_scores_diff = mastery_scores[0] - mastery_scores[1]
        mastery_scores_diff = mastery_scores_diff[np.newaxis]
        champion_team_masteries = champion_masteries_team(match, cursor)
        if champion_team_masteries is None:
            continue
        champion_team_diff = champion_team_masteries[0] - champion_team_masteries[1]
        champion_team_diff = champion_team_diff[np.newaxis]
        champion_summ_masteries = champion_masteries_summoner(match, cursor)
        if champion_summ_masteries is None:
            continue
        #historysumm = summoner_wins_and_rate_team(match, cursor)
        #historysumm_wins_diff = np.zeros(2)
        #historysumm_wins_diff[0] = historysumm[0] - historysumm[2]
        #historysumm_wins_diff[1] = historysumm[1] - historysumm[3]
        #historychamp = champion_wins_and_rate_team(match, cursor)
        #historychamp_wins_diff = np.zeros(2)
        #historychamp_wins_diff[0] = historychamp[0] - historychamp[2]
        #historychamp_wins_diff[1] = historychamp[1] - historychamp[3]

        #=================================IN==================================#  
        #zero_to_ = team_features_zero_to_thirty(match, cursor)
        #if zero_to_ is None:
        #    continue
        #zero_to_diff = zero_to_[:4] - zero_to_[4:]

        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        # PRE
        dataset[i] = np.concatenate((champions, spells, masteries, dmg_types, dmg_percent, mastery_scores, mastery_scores_diff, champion_team_masteries, champion_team_diff, champion_summ_masteries, winner))
#        dataset[i] = np.concatenate((272, 
        # IN
        #dataset[i] = np.concatenate((zero_to_, zero_to_diff, winner))
    dataset = remove_incomplete_instances(dataset)

    return dataset

#====================================MAIN=====================================#

def main(args):
    db = MySQLdb.connect(host="localhost", user="root", passwd="1234", 
                         db="lol")
    cursor = db.cursor()

    db.set_character_set('utf8')
    cursor.execute('SET NAMES utf8;')
    cursor.execute('SET CHARACTER SET utf8;')
    cursor.execute('SET character_set_connection=utf8;')

    model_name = args[0]

    feature_models = {"ft": feature_testing}
    model = feature_models[model_name](db, cursor)
    if model_name == "ft":
        model_name = args[1]
    np.savetxt(model_name + ".csv", model, delimiter=",", fmt="%.5g")

    cursor.close()
    db.close()


if __name__ == "__main__":
    main(sys.argv[1:])