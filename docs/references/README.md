# Key references

The papers closest to this project. The PDFs are kept locally only (git-ignored: publisher copyright).

| File | Paper | Why it matters here |
|---|---|---|
| `Joseph_Grunwald_Markov_Prefetching_retrospective.pdf` | D. Joseph and D. Grunwald, "Prefetching using Markov predictors - 25th anniversary retrospective," ISCA@50 retrospectives, 2 pages. | The authors' own short account of the Markov prefetcher our predictor is based on (why they built it, what it became). **Not the original paper**: cite the 1997 ISCA paper below. |
| `Wang_Canvas_NSDI2023.pdf` | C. Wang, Y. Qiao, H. Ma, S. Liu, Y. Zhang, W. Chen, R. Netravali, M. Kim and G. H. Xu, "Canvas: Isolated and adaptive swapping for multi-applications on remote memory," *Proc. USENIX NSDI*, 2023, pp. 161-179. | Kernel swap system (Linux) where each application gets its own prefetcher, adapted to its access pattern ("semantics-aware prefetching"). Closest to our *workload awareness*: one prefetching policy does not fit every workload, so adapt per workload at runtime. |
| `Akgun_KML_TOS2023.pdf` | I. U. Akgun, A. S. Aydin, A. Burford, M. McNeill, M. Arkhangelskiy and E. Zadok, "Improving storage systems using machine learning," *ACM Trans. Storage* 19(1), Art. 9, 30 pp., Jan. 2023. doi:10.1145/3568429 | A small ML model **inside the Linux kernel** that classifies the running workload and retunes the **readahead** size online (<4 KB memory, <0.2% CPU, up to 2.3x I/O throughput). Same idea as our runtime feedback deciding how aggressively to prefetch, and directly about readahead, our strongest baseline. Notes a misclassification case (reverse reads taken as sequential) worth citing next to our wrong-prefetch cost argument. |
| `Garg_CrossPrefetch_ASPLOS2024.pdf` | S. Garg, J. Zhang, R. Pitchumani, M. Parashar, B. Xie and S. Kannan, "CrossPrefetch: Accelerating I/O prefetching for modern storage," *Proc. ACM ASPLOS*, 2024, pp. 102-116. doi:10.1145/3617232.3624872 | Argues that OS prefetching is rigid because it never reports **prefetch effectiveness**; it exports cache state and prefetch effectiveness so a runtime can adapt how much it prefetches. That is the signal question our project studies (which feedback signal works). It does not limit the harm of wrong prefetches (pollution/eviction), which our decision engine does. File-level page cache, not anonymous memory paging. |

**Still cite the original Markov prefetcher:** D. Joseph and D. Grunwald, "Prefetching using Markov predictors," *Proc. ISCA*,
1997, pp. 252-263 (journal version: *IEEE Trans. Computers* 48(2):121-133, 1999). It has no free full-text copy; get it
through the VIT library (ACM Digital Library / IEEE Xplore).

Sources: USENIX open access (Canvas); Stony Brook FSL author copy (KML); NSF Public Access Repository author copy (CrossPrefetch).
