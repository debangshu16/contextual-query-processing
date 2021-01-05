from average_precision import ap1
from vsm import custom_vsm
import pandas as pd
from context_vsm import context_vsm
import sys
if __name__ == '__main__':

    include_context = int(sys.argv[1])
    #parameters
    retrieve_count = 3
    rel_level = 3
    #print (include_context)
    true_context = pd.read_csv('true_dataset_with_context.csv')

    data_file = 'data.txt'
    corpus = []
    with open(data_file,'r',encoding ='utf-8') as f:
        corpus = f.readlines()

    if not include_context:
        vsm = custom_vsm(corpus.copy(), use_idf = True, retrieve_count = retrieve_count)
    else:
        vsm = context_vsm(corpus.copy(), use_idf = False, retrieve_count = retrieve_count)

    queries = ['IIT cut off this year',  'current atmosphere pressure', 'air radio jockey',
                'computer science toppers this year', 'members of Air India club',
                'available luxury flight tickets'
                ]
    contexts = ['Rank',  'Nature', 'Broadcast',  'Rank', 'Flight', 'Flight']

    map1 = 0
    for query_id,query in enumerate(queries):
        print ("\nQuery \"{}\":\n".format(query))
        res = vsm.search(query)
        print ("Search Results:\n")
        retrieved = []


        for i in range(len(res)):
            docid, score = res[i][0], res[i][1]

            print ("Retrieved Document {}:\n{}".format((i+1),corpus[docid]), end = '')
            print ("Score:{}".format(score))

            t = true_context[true_context['DocID']==docid]

            c1 = t['Context'].values[0]

            retrieved.append(c1)



        ap = ap1(context = contexts[query_id], retrieved = retrieved,  rel_level = rel_level)
        print ("\nAverage precision at relevance level %d is %.3f" %(rel_level, ap))

        map1+=ap

    map1 = map1/len(queries)

    print ("\nMean Average Precision = %.3f" %map1)
