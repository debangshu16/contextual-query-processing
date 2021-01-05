def ap1(context, retrieved, rel_level):
    sap = 0

    for i in range(1, rel_level+1):
        sap += precision_at_k(retrieved, context, i)

    return (sap/rel_level)


def precision_at_k(retrieved,context,  k):
    c = 0
    matched = 0
    i = 0
    ap = 0
    n = len(retrieved)
    while matched<k and i<n:
        c+=1


        retrieved_context = retrieved[i]
        #print (retrieved_context)

        if retrieved_context == context:
            matched+=1
        if matched == k:
            ap+= matched/c

        i+=1

    return ap
