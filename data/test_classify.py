from keras.utils.np_utils import to_categorical
from random import sample

def seqfrmat(seqinp,maxlnpep):
    tmp=seqinp.strip()+"X"
    # print(tmp)
    while len(tmp)<=maxlnpep:
        tmp=tmp+" "
    coding=[]
    seqid=[]
    for x in range(0,maxlnpep+1):
        tmpctgr=to_categorical(aalist.index(tmp[x]), num_classes=len(aalist))
        # print(tmpctgr)
        coding.append(tmpctgr)
        seqid.append(aalist.index(tmp[x]))
    return seqid,coding


if __name__ == '__main__':
    maxlnpep=55
    aalist=["B","A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","X"," "]
    f=open('nonamp.csv','r')
    ln=f.readlines()[:]
    lenln=len(ln)
    clnpep=[]
    clncoding=[]
    f.close()
    seqlist=sample(range(0,lenln),lenln)
    print(seqlist)
    i = 0
    for i in range(0,lenln):
        if (len(ln[i])<=maxlnpep)&(i in seqlist):
            frmseq,frmcod=seqfrmat(ln[i],maxlnpep)
            frmcod=[[1]]
            i = i +1
            print(i)
            # print(frmseq)
            # print(frmcod)
            # clnpep.append(frmseq)
            # clncoding.append(frmcod)
        else:
            print(ln[i])

    # print(clnpep)


