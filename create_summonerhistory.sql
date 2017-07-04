USE lol;

CREATE TABLE SummonerChampHistory (summId INT(11), championId INT(11), wins INT(11), losses INT(11), rate FLOAT(5, 2), PRIMARY KEY(summId, championId), FOREIGN KEY(summId) REFERENCES Summoner(id) ON DELETE CASCADE, FOREIGN KEY(championId) REFERENCES Champion(id) ON DELETE CASCADE);
CREATE TABLE SummonerHistory (summId INT(11), wins INT(11), losses INT(11), rate FLOAT(5, 2), PRIMARY KEY(summId), FOREIGN KEY(summId) REFERENCES Summoner(id) ON DELETE CASCADE);