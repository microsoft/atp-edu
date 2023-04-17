from datetime import datetime

import pandas as pd
import numpy as np
import json
import math


class DEKnowledge:
    def __init__(self):
        return

    def __init__(self, id, type, regions, standardQuestion, extendedQuestions, validTime, expiredTime, isPublished):
        self.id = id
        self.type = type
        self.regions = regions.split('|')
        self.standardQuestion = standardQuestion
        self.extendedQuestions = extendedQuestions
        self.validTime = validTime
        self.expiredTime = expiredTime
        self.isPublished =  True if isPublished == '是' else False
        return

    def extendInfo(self, channelStr, recommendedQuestions, message):
        self.channels = channelStr.split('|')
        self.recommendedQuestions = recommendedQuestions
        self.response = message
        return

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4, ensure_ascii=False)


class KSParser:
    def __init__(self):
        return
    def parse(self, excel_path):
        df = pd.read_excel(excel_path, sheet_name="Sheet1", skiprows=1)
        knowledges = []
        for idx, row in df.iterrows():

            if row["知识点ID"] is not np.nan:
                extendedQuestionStr = row["扩展问(多个扩展问则显示为多行)"]
                extendedQuestions = [] if extendedQuestionStr is np.nan else str(extendedQuestionStr).split('\n')

                knowledge = DEKnowledge(row["知识点ID"], row["多层级的问题类别(由 \"|\" 分开)"],
                                        row["地区(多个地区则用 \"|\" 分开)"], row["标准问"], extendedQuestions,
                                        row["生效时间"], row["失效时间"], row["是否发布"])
                knowledges.append(knowledge)
                currentKnowledge = knowledge
            else:
                recommendedQuestions = []
                if row["推荐问(多个推荐问则显示为多行)"] is not np.nan and str(row["推荐问(多个推荐问则显示为多行)"]) != "nan":
                    recommendedQuestions = str(row["推荐问(多个推荐问则显示为多行)"]).split('\n')
                currentKnowledge.extendInfo(row["渠道(多个渠道支持则用 \"|\" 分开)"], recommendedQuestions, row["消息1"])

        return knowledges


if __name__ == "__main__":
    path = "C:\\Users\\jull\\Downloads\\Knowledge\\faqTemplate.xlsx"
    parser = KSParser()
    parser.parse(path)