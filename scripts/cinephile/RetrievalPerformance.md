# Retrieval Performance

The organisers of the [Cinephile-2005 challenge](https://hermes-hub.de/forschen/datachallenges/challenges/challenge-2025.html) have released the [ground_truth_validation_dataset.json](https://thor.robots.ox.ac.uk/wise/assets/cinephile/ground_truth_validation_dataset.json) file that contains 22 search queries and a list of video filenames that should be retrieved by a video retrieval system as the top matching results. The [`evaluate-performance.py`](evaluate-performance.py) script is used to compute the number of videos contained in the ground truth (i.e. the JSON file) that can be correctly retrieved the [WISE Search Engine (WISE)](https://meru.robots.ox.ac.uk/cinephile/).

```
$ python scripts/cinephile/evaluate-performance.py \
  --wise-url https://meru.robots.ox.ac.uk/cinephile/ \
  --ground-truth-url https://thor.robots.ox.ac.uk/wise/assets/cinephile/ground_truth_validation_dataset.json

...
|-------+-------------------------------------------------+--------|
| top-k | Update to the original query                    | Recall |
|-------+-------------------------------------------------+--------|
| 1000  | Find videos -> Old photos                       | 0.47   |
| 1000  | Find videos -> Photo                            | 0.42   |
| 1000  | Find videos -> ""                               | 0.42   |
| 1000  | Find videos -> German and Dutch archival photos | 0.40   |
| 1000  | None (i.e. original query was used as it is)    | 0.40   |
| 1000  | videos -> photos                                | 0.39   |
| 1000  | Find videos -> Old video frames                 | 0.38   |
| 1000  | videos -> images                                | 0.35   |
|-------+-------------------------------------------------+--------|
```

A query with `recall=1.0` corresponds to the case when all the video filenames contained in the ground truth JSON file is retrieved by the WISE search engine. This indicates that the search engine is successful in retrieving all the videos which are known to contain the scene described by the search query. Similarity, a query with `recall=0.0` indicates that the WISE search engine was not able to retrieve any of the video files known to contain the scene describe by the search query. We group the search queries based on their recall values and analyse their performance below. The image and text features were extracted using the [`mlfoundations/open_clip/ViT-L-16-SigLIP2-512/webli`](https://github.com/mlfoundations/open_clip/) vision language model. The search result rank (or, position in ordered search results) is shown in `[rank]` (e.g. [8]). For all the 22 queries, we replaced occurrances of prefix `Find videos` with `Old photos` as this improves the retrieval performance as shown below.

## Queries for which recall is 1.0
- Old photos shot at a train station
    - [8] [DFF_KriegsgefangeneFranzosenI_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/740) ([33.00s - 54.21s](https://meru.robots.ox.ac.uk/cinephile/media/740#t=33.0,54.21))
    - [49] [AANKOMST CHARLES KING MET VROUW EN KINDEREN.mp4](https://meru.robots.ox.ac.uk/cinephile/media/62) (0.52s - 12.16s)
    - [53] [DFF_EinzugDerUnteroffizierschule_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/787) ([173.38s - 207.33s](https://meru.robots.ox.ac.uk/cinephile/media/787#t=173.38,207.33))
    - [63] [Officiele opening van de geelectrificeerde lijn Eindhoven-Maastricht.mp4](https://meru.robots.ox.ac.uk/cinephile/media/98) ([8.64s - 16.24s](https://meru.robots.ox.ac.uk/cinephile/media/98#t=8.64,16.24))
    - [91] [DFF_KriegsgefangeneFranzosenIVKriegsgefangeneTurkosInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/742) ([48.00s - 79.83s](https://meru.robots.ox.ac.uk/cinephile/media/742#t=48.00,79.83))

- Old photos that show industry
    - [44] [DFF_ErnstLeitzOptischeWerkeWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/669) ([127.96s - 143.71s](https://meru.robots.ox.ac.uk/cinephile/media/669#t=127.96,143.71))
    - [136] [DFF_KriegseinsatzDerFrauen__Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/733) ([65.71s - 81.58s](https://meru.robots.ox.ac.uk/cinephile/media/733#t=65.71,81.58))

- Old photos that contain tracking shots
    - [94] [AAPJES EN AAPJES.mp4](https://meru.robots.ox.ac.uk/cinephile/media/299) ([29.72s - 33.12s](https://meru.robots.ox.ac.uk/cinephile/media/299#t=29.72,33.12))

- Old photos containing closeups of machines
    - [1] [DFF_ErnstLeitzOptischeWerkeWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/669) ([147.12s - 179.96s](https://meru.robots.ox.ac.uk/cinephile/media/669#t=147.12,179.96))
    - [18] [ARNHEM KAN WEER AUTOMATISCH TELEFONEREN.mp4](https://meru.robots.ox.ac.uk/cinephile/media/157) ([71.20s - 73.60s](https://meru.robots.ox.ac.uk/cinephile/media/157#t=71.20,73.60))
    - [20] [DFF_Montage_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/667) ([160.67s - 179.67s](https://meru.robots.ox.ac.uk/cinephile/media/667#t=160.67,179.67))

## Queries for which recall is between 0.7 and 1.0
- Old photos showing people riding a boat ( Recall= 0.75 )
    - [New stunt world record on the water.mp4](https://meru.robots.ox.ac.uk/cinephile/media/223) (**Failed-to-Retrieve**)
    - [1] [DFF_Ruderregatta_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/728) ([160.83s - 207.42s](https://meru.robots.ox.ac.uk/cinephile/media/728#t=160.83,207.42))
    - [5] [DFF_HochwasserInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/696) ([158.38s - 172.42s](https://meru.robots.ox.ac.uk/cinephile/media/696#t=158.38,172.42))
    - [33] [DFF_Bootsfahrt_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/753) ([59.29s - 93.88s](https://meru.robots.ox.ac.uk/cinephile/media/753#t=59.29,93.88))

- Old photos with animals ( Recall= 0.83 )
    - [Dolfinarium uitgebreid met walrussenbassin.mp4](https://meru.robots.ox.ac.uk/cinephile/media/53) (**Failed-to-Retrieve**)
    - [33] [Oude man met hond en gans.mp4](https://meru.robots.ox.ac.uk/cinephile/media/502) ([10.60s - 16.84s](https://meru.robots.ox.ac.uk/cinephile/media/502#t=10.60,16.84))
    - [52] [DFF_OchsenfestTierschaufestInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/672) ([589.29s - 616.75s](https://meru.robots.ox.ac.uk/cinephile/media/672#t=589.29,616.75))
    - [57] [AAPJES EN AAPJES.mp4](https://meru.robots.ox.ac.uk/cinephile/media/299) ([5.04s - 11.44s](https://meru.robots.ox.ac.uk/cinephile/media/299#t=5.04,11.44))
    - [100] [DFF_BlutwaescheBeiEinemHund_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/768) ([263.96s - 264.58s](https://meru.robots.ox.ac.uk/cinephile/media/768#t=263.96,264.58))
    - [174] [INTERNATIONALE HONDENTENTOONSTELLING.mp4](https://meru.robots.ox.ac.uk/cinephile/media/622) ([28.52s - 31.44s](https://meru.robots.ox.ac.uk/cinephile/media/622#t=28.52,31.44))

- Old photos with humans working with machines ( Recall= 0.8 )
    - [ARNHEM KAN WEER AUTOMATISCH TELEFONEREN.mp4](https://meru.robots.ox.ac.uk/cinephile/media/157) (**Failed-to-Retrieve**)
    - [1] [DFF_ErnstLeitzOptischeWerkeWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/669) ([147.12s - 179.96s](https://meru.robots.ox.ac.uk/cinephile/media/669#t=147.12,179.96))
    - [2] [Wanderung in einer mechan. Werkstätte.mp4](https://meru.robots.ox.ac.uk/cinephile/media/716) ([61.04s - 64.17s](https://meru.robots.ox.ac.uk/cinephile/media/716#t=61.04,64.17))
    - [14] [DFF_KriegseinsatzDerFrauen__Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/733) ([81.62s - 102.79s](https://meru.robots.ox.ac.uk/cinephile/media/733#t=81.62,102.79))
    - [26] [DFF_Montage_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/667) ([114.88s - 127.54s](https://meru.robots.ox.ac.uk/cinephile/media/667#t=114.88,127.54))

## Queries for which recall is between 0.3 and 0.7
- Old photos containing the medium shot of a woman ( Recall= 0.5 )
    - [ATELIER HOLLANDIA STAVES BIOS AND DEAN.mp4](https://meru.robots.ox.ac.uk/cinephile/media/253) (**Failed-to-Retrieve**)
    - [4] [DFF_Portraits_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/704) ([70.71s - 104.08s](https://meru.robots.ox.ac.uk/cinephile/media/704#t=70.71,104.08))

- Old photos in which people are dancing ( Recall= 0.67 )
    - [DFF_Eislaeufer_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/703) (**Failed-to-Retrieve**)
    - [4] [ÉÉN MEI DAG.mp4](https://meru.robots.ox.ac.uk/cinephile/media/126) ([20.76s - 34.28s](https://meru.robots.ox.ac.uk/cinephile/media/126#t=20.76,34.28))
    - [18] [KINDERBALLET DANST DE &quot;QUADRILLE LANCIERS&quot;.mp4](https://meru.robots.ox.ac.uk/cinephile/media/267) ([31.84s - 36.60s](https://meru.robots.ox.ac.uk/cinephile/media/267#t=31.84,36.60))

- Old photos showing artillery or war machinery ( Recall= 0.5 )
    - Messter-Woche Italienischer Kriegsschauplatz.mp4 (**Failed-to-Retrieve**)
    - [Woevre-Städte als Opfer der französischen Artillerie.mp4](https://meru.robots.ox.ac.uk/cinephile/media/713) (**Failed-to-Retrieve**)
    - [4] [Kraftwagen Flaks.mp4](https://meru.robots.ox.ac.uk/cinephile/media/690) ([268.98s - 397.98s](https://meru.robots.ox.ac.uk/cinephile/media/690#t=268.98,397.98))
    - [80] [Kriegsflieger an der Westfront.mp4](https://meru.robots.ox.ac.uk/cinephile/media/750) ([740.74s - 776.99s](https://meru.robots.ox.ac.uk/cinephile/media/750#t=740.74,776.99))

- Old photos shwoing a group of people in full shot ( Recall= 0.6 )
    - [DFF_ErnstLeitzOptischeWerkeWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/669) (**Failed-to-Retrieve**)
    - [10-jarig bestaan van het regiment motorartillerie.mp4](https://meru.robots.ox.ac.uk/cinephile/media/626) (**Failed-to-Retrieve**)
    - [30] [DFF_HochwasserInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/696) ([352.25s - 379.71s](https://meru.robots.ox.ac.uk/cinephile/media/696#t=352.25,379.71))
    - [57] [DFF_WassersportKopfspruengeMitGLeitz_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/749) ([88.00s - 99.79s](https://meru.robots.ox.ac.uk/cinephile/media/749#t=88.00,99.79))
    - [86] [DFF_ImFreibadAnDerLahn_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/743) ([36.42s - 47.75s](https://meru.robots.ox.ac.uk/cinephile/media/743#t=36.42,47.75))

- Old photos that show a city ( Recall= 0.33 )
    - [DFF_ArbeiterVerlassenDasLeitzwerk_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/747) (**Failed-to-Retrieve**)
    - [DFF_KriegsendeInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/666) (**Failed-to-Retrieve**)
    - [5] [DFF_LahntalBadEms_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/736) ([57.25s - 136.08s](https://meru.robots.ox.ac.uk/cinephile/media/736#t=57.25,136.08))

- Old photos showing a group of civilians ( Recall= 0.22 )
    - [DFF_ArbeiterVerlassenDasLeitzwerk_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/747) (**Failed-to-Retrieve**)
    - [DFF_EinzugDerUnteroffizierschule_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/787) (**Failed-to-Retrieve**)
    - [DFF_HochwasserInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/696) (**Failed-to-Retrieve**)
    - [DFF_ImFreibadAnDerLahn_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/743) (**Failed-to-Retrieve**)
    - [DFF_Karneval_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/782) (**Failed-to-Retrieve**)
    - [OOGSTFEEST.mp4](https://meru.robots.ox.ac.uk/cinephile/media/597) (**Failed-to-Retrieve**)
    - [ÉÉN MEI DAG.mp4](https://meru.robots.ox.ac.uk/cinephile/media/126) (**Failed-to-Retrieve**)
    - [64] [DFF_OchsenfestTierschaufestInWetzlar_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/672) ([107.88s - 111.29s](https://meru.robots.ox.ac.uk/cinephile/media/672#t=107.88,111.29))
    - [114] [DFF_ErholungsheimOberjosbach_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/699) ([86.71s - 125.96s](https://meru.robots.ox.ac.uk/cinephile/media/699#t=86.71,125.96))

## Queries for which recall is 0.0
- Old photos containing closeups of humans
    - [Grandpa Klijzing, world champion pipe smoking.mp4](https://meru.robots.ox.ac.uk/cinephile/media/592)

- Old photos containing reverse motion
    - [DFF_WassersportKopfspruengeMitGLeitz_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/749)

- Old photos that contain pans
    - [DFF_EinzugDerUnteroffizierschule_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/787)
    - [DFF_LahntalBadEms_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/736)
    - [Expansion of Technical College.mp4](https://meru.robots.ox.ac.uk/cinephile/media/12)
    - [Woevre-Städte als Opfer der französischen Artillerie.mp4](https://meru.robots.ox.ac.uk/cinephile/media/713)

- Old photos that contain tilts
    - [Agreement on Scheldt-Rhine connection.mp4](https://meru.robots.ox.ac.uk/cinephile/media/520)
    - [Expansion of Technical College.mp4](https://meru.robots.ox.ac.uk/cinephile/media/12)

- Old photos that contain two shots
    - [ATELIER HOLLANDIA STAVES BIOS AND DEAN.mp4](https://meru.robots.ox.ac.uk/cinephile/media/253)
    - [DFF_Bootsfahrt_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/753)
    - [DFF_ErholungsheimOberjosbach_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/699)

- Old photos that contain zoom outs
    - [Demonstratie brandvrije benzinetank.mp4](https://meru.robots.ox.ac.uk/cinephile/media/337)

- Old photos that show groups of soldiers in medium shot
    - [DFF_EinzugDerUnteroffizierschule_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/787)

- Old photos that show naked people
    - [DFF_EinzugDerUnteroffizierschule_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/787)

- Old photos with bird's-eye perspecitve
    - [DFF_ArbeiterVerlassenDasLeitzwerk_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/747)
    - [DFF_Karneval_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/782)
    - [DFF_LahntalBadEms_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/736)
    - [DFF_UkrainerInWetzlarFahnenweiheUndHeimkehr_Logo_1080p_H264.mp4](https://meru.robots.ox.ac.uk/cinephile/media/674)
    - [Einzug des siegreichen Generals von Mannerheim in Helsingfors.mp4](https://meru.robots.ox.ac.uk/cinephile/media/691)
    - [OOGSTFEEST.mp4](https://meru.robots.ox.ac.uk/cinephile/media/597)
