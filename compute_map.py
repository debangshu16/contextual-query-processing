import pandas as pd
def ap1(context, retrieved, rel_level):
    sap = 0

    for i in range(1, rel_level+1):
        sap += precision_at_k(retrieved, context, i)

    return (sap/rel_level)

def get_context(doc_id):
    #return true_context[doc_id]
    t = true_context[true_context['DocID']==doc_id]
    retrieved_context = t['Context'].values[0]
    return retrieved_context

def precision_at_k(retrieved,context, k):
    c = 0
    matched = 0
    i = 0
    ap = 0
    n = len(retrieved)
    while matched<k and i<n:
        c+=1

        doc = retrieved[i]
        #t = true_context[true_context['doc']==doc]
        #retrieved_context = t['context']
        retrieved_context = get_context(doc)

        if retrieved_context == context:
            matched+=1
        if matched == k:
            ap+= matched/c

        i+=1

    return ap

#true_context = pd.read_csv('true_dataset_with_context.csv')
'''if __name__== '__main__':
    #query = ''
    true_context = [1,1,1,1,0,1,0,1,0,1]
    #print ("Map = %f" %ap1(retrieved = [1,2,4,5,6,8], context = 0, rel_level = 3))
    #true_context = pd.read_csv('true_dataset_with_context.csv')

    r1 = [1,2,4,5,6,8]
    c1 = 0

    r2 = [4,5,6,7,8,9]
    c2 = 1

    r3 = [3,4,5,6,7,8]
    c3 = 0

    map1 = 0
    returned = [r1,r2,r3]
    contexts = [c1,c2,c3]

    n = len(returned)
    for i in range(n):
        map1 += ap1(retrieved = returned[i], context = contexts[i], rel_level = 3)

    map1 = map1/n
    print (map1)'''
