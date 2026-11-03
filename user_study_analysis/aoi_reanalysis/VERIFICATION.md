# v6 verification

Canonical hierarchy:

1. Self + Partner + Information + Other = 100%.
2. H1-H4 subdivide only the already-allocated Information dwell.
3. Therefore H1 + H2 + H3 + H4 = Information in every H4 trial (floating-point tolerance).

## Actual uploaded-results verification

Primary H1-H4 descriptive means (% total fixation):

FutureStep         1         2         3         4
Time                                              
200         2.895039  3.033422  2.517868  2.216430
400         3.643700  3.246032  2.688527  2.221942
600         4.052351  3.469971  2.839786  2.379952

Primary hierarchical H1-H4 LMM target tests:

           term  Wald_chi2  df      p_value              random_effects
           Time  18.728828   2 8.572089e-05 (1|Participant) + (1|Trial)
     FutureStep 259.019713   3 7.325843e-56 (1|Participant) + (1|Trial)
Time×FutureStep  14.281506   6 2.664488e-02 (1|Participant) + (1|Trial)

Relative-allocation LMM target tests:

           term  Wald_chi2  df      p_value  random_effects
           Time   1.992962   2 3.691762e-01 (1|Participant)
     FutureStep 109.975572   3 1.110689e-23 (1|Participant)
Time×FutureStep  11.435708   6 7.580832e-02 (1|Participant)

Near/Far hierarchical LMM target tests:

      term  Wald_chi2  df      p_value              random_effects
      Time  16.744440   2 2.312017e-04 (1|Participant) + (1|Trial)
     Range 131.642919   1 1.791043e-30 (1|Participant) + (1|Trial)
Time×Range   2.397823   2 3.015223e-01 (1|Participant) + (1|Trial)

Final validator: PASS

Unit tests: 17 passed.
