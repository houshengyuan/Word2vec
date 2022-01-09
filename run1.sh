#!/usr/bin/env bash

python main.py --task cbow --hierarchical ;

#Token number: 205068
#Vocab size: 17832
#Cost 36.5 min
#[('i', 1.0), ('prove', 0.8298406399772222), ('niccol', 0.826815637071844), ('crave', 0.8259397401690036), ('reunions', 0.8215614258907181), ('unexplainable', 0.8206342743392268), ('austen'$ 0.8199281437827426), ('dressing', 0.7880784602188118), ('muster', 0.7826751910237786), ('chronic', 0.7815157302107972)]
#Load model from tmp/ckpt
#spearman correlation: 0.298
#pearson correlation: 0.350

python main.py --task skip-gram --hierarchical ;

#Token number: 205068
#Vocab size: 17832
#Cost 44.0 min
#[('i', 1.0), ('was', 0.8911930839697859), ('have', 0.8880309430882156), ('possibilities', 0.8776109260689164), ('highest', 0.867047202154738), ('would', 0.8325100860545628), ('duke', 0.832$067199191462), ('bullets', 0.8293291074211881), ('underdogs', 0.8281653385881208), ('makin', 0.826301490991854)]
#Load model from tmp/ckpt
#spearman correlation: 0.259
#pearson correlation: 0.278

python main.py --task cbow ;

#Token number: 205068
#Vocab size: 17832
#Cost 211.7 min
#[('i', 1.0), ('we', 0.912838794610602), ('you', 0.8877629189472462), ('handed', 0.8714395511225913), ('average', 0.8636565282360613), ('fundamental', 0.8626562376034204), ('been', 0.846205$764873658), ('tasteless', 0.8432406953471567), ('fresnadillo', 0.8408071660012586), ('unerring', 0.8374040713583903)]
#Load model from tmp/ckpt
#spearman correlation: 0.366
#pearson correlation: 0.439

python main.py --task skip-gram ;

#Token number: 205068
#Vocab size: 17832
#Cost 108.3 min
#[('i', 0.9999999999999999), ('might', 0.9835428188002951), ('can', 0.9762453600282223), ('d', 0.9695785412389974), ('been', 0.9689148116972351), ('really', 0.9686983604420812), ('have', 0.9672876046621678), ('you', 0.9666020558525854), ('we', 0.9662675313113993), ('do', 0.9656380926535335)]
#Load model from tmp/ckpt
#spearman correlation: 0.280
#pearson correlation: 0.328

python main.py --task cbow --neg --sample-size 5 ;

#Token number: 205068
#Vocab size: 17832
#Cost 36.8 min
#[('i', 1.0), ('we', 0.9838933757494994), ('you', 0.9719359191782915), ('really', 0.9397979019982571), ('they', 0.9334364017041251), ('only', 0.9236551119389811), ('can', 0.9212328629523637), ('just', 0.9141594592357905), ('not', 0.904173594345997), ('will', 0.8996711897522732)]
#Load model from tmp/ckpt
#spearman correlation: 0.315
#pearson correlation: 0.373

python main.py --task skip-gram --neg --sample-size 5 ;

#Token number: 205068
#Vocab size: 17832
#Cost 36.7 min
#[('i', 1.0000000000000002), ('if', 0.9959796081131643), ('you', 0.9945359197681375), ('have', 0.9931985196680171), ('don', 0.9924851230467084), ('can', 0.9910687943533979), ('ll', 0.9897596564128698), ('isn', 0.9893464014980207), ('when', 0.9876310959446541), ('see', 0.9874353138429615)]
#Load model from tmp/ckpt
#spearman correlation: 0.275
#pearson correlation: 0.327

python main.py --task cbow --neg --sample-size 5 --sub-sampling --subsample-thr 1e-3;

#Token number: 205068
#Vocab size: 17832
#Cost 37.8 min
#[('i', 0.9999999999999999), ('you', 0.9945877939777044), ('of', 0.9935617146926743), ('to', 0.9927375271004053), ('just', 0.9921634180156711), ('not', 0.9919653469676359), ('can', 0.9912873315121024), ('we', 0.9906736802783208), ('the', 0.9905759012174058), ('one', 0.9901646074389117)]
#Load model from tmp/ckpt
#spearman correlation: 0.295
#pearson correlation: 0.360

python main.py --task skip-gram --neg --sample-size 5 --sub-sampling --subsample-thr 1e-3;

#Token number: 205068
#Vocab size: 17832
#Cost 38.8 min
#[('i', 1.0), ('ve', 0.9991217978756416), ('ll', 0.9985940738868997), ('one', 0.9985585927101), ('character', 0.9985139754141513), ('also', 0.9984697055234741), ('kind', 0.9982921927186703), ('make', 0.9981991117104634), ('all', 0.9981539357511393), ('things', 0.9981527210454854)]
#Load model from tmp/ckpt
#spearman correlation: 0.244
#pearson correlation: 0.282




