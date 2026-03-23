from vae_dec import dec_util
if __name__=="__main__":

    dec_util.train_dec('./raw/SRR13128012.fastq','./SRR13128012', 4,[128,128],5,30000)

    #dec_util.test_dec( 4,[128,128],10,'cuda')