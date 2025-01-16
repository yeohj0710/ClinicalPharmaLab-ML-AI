$PARAM
TVCL = 0.0094
TVV1 = 3.63
TVV2 = 2.78
TVQ = 0.0321
TVEMAX = -0.295
BW = 109.14

EGFR = 112.2

BPS = 0

SEX = 0

RAAS = 0


$MAIN
double CL = TVCL * pow(BW / 80, 0.566) * pow(EGFR / 90, 0.186) * pow(exp(0.172), BPS) * pow(exp(0.165), SEX) * pow(exp(-0.125), RAAS) * (exp((TVEMAX * pow(TIME, 3.15)) / (pow(1410, 3.15) + pow(TIME, 3.15))) + EEMAX) * exp(ECL);
double V1 = TVV1 * pow(BW / 80, 0.597) * pow(exp(0.152), SEX) * exp(EV1);
double V2 = TVV2 * exp(EV2);
double Q = TVQ;

$OMEGA
@labels EEMAX
0.0719

$OMEGA
@labels ECL EV1 EV2
@cor
0.123
0.0432 0.123
0 0 0.258


$SIGMA 0

$PKMODEL cmt = "CENT, PERIPH", depot = FALSE

$POST
capture CP = (CENT / V1) * (1 + EPS(1));
