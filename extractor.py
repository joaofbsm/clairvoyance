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

    blue_champions = np.zeros(len(champions["pos"]), dtype="int")
    red_champions = np.zeros(len(champions["pos"]), dtype="int")

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

    blue_spells = np.zeros(len(spells["pos"]), dtype="int")
    red_spells = np.zeros(len(spells["pos"]), dtype="int")

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
    mastery_scores = np.zeros(2, dtype="int")
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)[0][0]
        mastery_scores[0] += mastery_score


    for _, player in red_team.iterrows():
        cursor.execute(get_mastery_scores, [player["summonerId"]])
        mastery_score = list(cursor)[0][0]
        mastery_scores[1] += mastery_score

    return mastery_scores


def champion_masteries_team(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    champion_masteries = np.zeros(2, dtype="int")
    
    for _, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        champion_masteries[0] += champion_mastery


    for _, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        champion_masteries[1] += champion_mastery

    return champion_masteries


def champion_masteries_summoner(match, cursor):
    get_champion_masteries = ("SELECT mastery "
                              "FROM SummonerChampMasteries "
                              "WHERE summId = %s AND championId = %s")

    blue_team = match[:5]
    red_team = match[5:10]
    blue_champion_masteries = np.zeros(5, dtype="int")
    red_champion_masteries = np.zeros(5, dtype="int")
    
    for i, player in blue_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        blue_champion_masteries[i] = champion_mastery


    for i, player in red_team.iterrows():
        cursor.execute(get_champion_masteries, (player["summonerId"],
                                                player["championId"]))
        champion_mastery = list(cursor)[0][0]
        red_champion_masteries[i] = champion_mastery

    champion_masteries = np.concatenate((blue_champion_masteries, 
                                         red_champion_masteries))

    return champion_masteries

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


def remove_incomplete_instances(dataset):
    incomplete_instances = []
    for i, instance in enumerate(dataset):
        complete = np.count_nonzero(instance)
        if not complete:
            incomplete_instances.append(i)

    print("\n\n", len(incomplete_instances), "incomplete instances:\n\n", 
          incomplete_instances)
    dataset = np.delete(dataset, incomplete_instances, axis=0)

    return dataset

#===================================MODELS====================================#

def build_model_pre1(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 273))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, winner))

    return dataset   


def build_model_pre2(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " P.spell1Id, P.spell2Id, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 291))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        spells = onehot_spells(match, db)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, spells, winner))

    return dataset 


def build_model_pre3(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 363))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        team_masteries = onehot_summoner_masteries_team(match, db, cursor)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, team_masteries, winner))

    return dataset 


def build_model_pre4(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " P.spell1Id, P.spell2Id, T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 381))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        spells = spells = onehot_spells(match, db)
        team_masteries = onehot_summoner_masteries_team(match, db, cursor)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, spells, team_masteries, 
                                     winner))

    return dataset 


def build_model_pre5(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 279))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        dmg_types = dmg_types_team(match, db)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, dmg_types, winner))

    return dataset   


def build_model_pre6(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 277))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        dmg_types_percent = dmg_types_percent_team(match, db)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((champions, dmg_types_percent, winner))

    return dataset   


def build_model_pre7(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 275))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        mastery_scores = mastery_scores_team(match, cursor)
        #mastery_scores_diff = mastery_scores[0] - mastery_scores[1]
        winner = np.array(df["winner"].iloc[player])[np.newaxis]
        dataset[i] = np.concatenate((champions, mastery_scores, winner))

    return dataset


def build_model_pre8(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 275))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        champion_masteries = champion_masteries_team(match, cursor)
       #champion_masteries_diff = champion_masteries[0] - champion_masteries[1]
        winner = np.array(df["winner"].iloc[player])[np.newaxis]
        dataset[i] = np.concatenate((champions, champion_masteries, winner))

    return dataset


def build_model_pre9(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 283))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        champion_masteries = champion_masteries_summoner(match, cursor)
        winner = np.array(df["winner"].iloc[player])[np.newaxis]
        dataset[i] = np.concatenate((champions, champion_masteries, winner))

    return dataset


def build_model_pre_in1(db, cursor):
    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 281))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        zero_to_ten = team_features_zero_to_ten(match, cursor)
        if zero_to_ten is None:
            continue
        #zero_to_ten_diff = zero_to_ten[:4] - zero_to_ten[4:]
        winner = np.array(df["winner"].iloc[player])[np.newaxis]
        dataset[i] = np.concatenate((champions, zero_to_ten, winner))

    dataset = remove_incomplete_instances(dataset)

    return dataset


def build_model_pre_in1_all(db, cursor):
    """In attributes + diff."""


    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 285))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        champions = onehot_champions(match, db)
        zero_to_ten = team_features_zero_to_ten(match, cursor)
        if zero_to_ten is None:
            continue
        zero_to_ten_diff = zero_to_ten[:4] - zero_to_ten[4:]
        winner = np.array(df["winner"].iloc[player])[np.newaxis]
        dataset[i] = np.concatenate((champions, zero_to_ten, zero_to_ten_diff,
                                     winner))

    dataset = remove_incomplete_instances(dataset)

    return dataset

def feature_testing(db, cursor):
    """
    Size of features
    ----------------

    onehot_champions: 272
    mastery_scores_team = 2
    mastery_scores_diff = 1
    zero_to_ten = 8
    zero_to_ten_diff = 4
    winner: 1

    TOTAL: 288
    """

    df = pd.read_sql("SELECT D.matchId, PL.summonerId, P.championId, P.teamId,"
                     " T.winner "
                     "FROM MatchParticipant P, MatchDetail D, MatchTeam T, "
                     "MatchPlayer PL "
                     "WHERE P._match_id = D.matchId AND D.mapId = 11 "
                     "AND D.matchId = T._match_id AND P.teamId = T.teamId "
                     "AND PL._participant_id = P._id "
                     "ORDER BY D.matchId, P.teamId ", db)

    dataset = np.zeros((df.shape[0] / 10, 16))
    bar = tqdm(total=df.shape[0] / 10)
    for i, player in enumerate(xrange(0, df.shape[0] - 10, 10)):
        bar.update(1)
        match = df[player:player + 10]

        #champions = onehot_champions(match, db)

        mastery_scores = mastery_scores_team(match, cursor)
        mastery_scores_diff = mastery_scores[0] - mastery_scores[1]
        mastery_scores_diff = mastery_scores_diff[np.newaxis]

        zero_to_ten = team_features_zero_to_ten(match, cursor)
        if zero_to_ten is None:
            continue
        zero_to_ten_diff = zero_to_ten[:4] - zero_to_ten[4:]

        winner = np.array(df["winner"].iloc[player])[np.newaxis]

        dataset[i] = np.concatenate((mastery_scores, mastery_scores_diff, zero_to_ten, zero_to_ten_diff, winner))

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

    feature_models = {"pre1": build_model_pre1,
                      "pre2": build_model_pre2,
                      "pre3": build_model_pre3,
                      "pre4": build_model_pre4,
                      "pre5": build_model_pre5,
                      "pre6": build_model_pre6,
                      "pre7": build_model_pre7,
                      "pre8": build_model_pre8,
                      "pre9": build_model_pre9,
                      "prein1": build_model_pre_in1,
                      "prein1all": build_model_pre_in1_all,
                      "feat_test": feature_testing}
    model = feature_models[model_name](db, cursor)
    np.savetxt(model_name + ".csv", model, delimiter=",", fmt="%.5g")

    cursor.close()
    db.close()


if __name__ == "__main__":
    main(sys.argv[1:])