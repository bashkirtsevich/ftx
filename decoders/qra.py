import typing
import json
import numpy as np
from numba import njit
import numpy.typing as npt

from consts.q65 import *
from ldpc.qra_decoder import q65_intrinsics_ff, q65_dec_fullaplist, q65_dec, q65_init


def smo121(a: np.ndarray):
    v = np.array([0.25, 0.5, 0.25])
    convolved = np.convolve(a, v, mode='same')
    return np.concat([a[:1], convolved[1:-1], a[-1:]])


def dB(x: float) -> float:
    val = -99.0
    if x > 1.259e-10:
        if x < 0.000001:
            x = 0.000001
        val = 10.0 * np.log10(x)
    return val


# @njit
def shell(arr):
    n = len(arr)
    inc = 1
    while 3 * inc + 1 <= n:
        inc = 3 * inc + 1

    while inc > 1:
        for i in range(inc, n):
            v = arr[i]
            j = i
            while j >= inc and arr[j - inc] > v:
                arr[j] = arr[j - inc]
                j -= inc
            arr[j] = v
        inc //= 3

    return arr


# @njit
def q65_bzap(s3f: npt.NDArray[np.float64], LL: int):
    NBZAP = 15
    hist = np.zeros(LL, dtype=np.int64)

    for j in range(63):
        beg = j * LL
        ipk1 = np.argmax(s3f[beg:beg + LL])
        hist[ipk1] += 1

    if np.max(hist) > NBZAP:
        for i in range(LL):
            if hist[i] > NBZAP:
                for j in range(63):
                    s3f[j * LL + i] = 1.0


NMAX_PT = 141072


# @njit
def pctile_shell(x: npt.NDArray[np.float64], npts: int, npct: int) -> float:
    xpct = 1.0

    if npts <= 0:
        xpct = 1.0
        return xpct

    if npts > NMAX_PT:
        return xpct

    pctile_shell_tmp = np.zeros(npts, dtype=np.float64)
    pctile_shell_tmp[:] = x[:npts]
    pctile_shell_tmp = shell(pctile_shell_tmp)

    j = int(npts * 0.01 * npct)

    if j < 0:
        j = 0

    if j > npts - 1:
        j = npts - 1

    return pctile_shell_tmp[j]


class Q65:
    NSTEP = 8
    NMAX = (120 * 12000)  # !Max TRperiod is 300 s
    PLOG_MIN = -242.0  # !List decoding threshold
    MAX_NCW = 206
    MSX_DEC = 100
    MSX_NQF = 20

    ccf_offsetr = 70
    ccf_offsetc = 7000

    mode_q65 = 2
    nsps = 3600
    npasses = 2
    istep = int(nsps / NSTEP)
    dtstep = nsps / (NSTEP * 12000.0)
    lag1 = int(-1.0 / dtstep)
    lag2 = int(1.0 / dtstep + 0.9999)

    i0 = 0
    j0 = 0

    LL0 = 0
    iz0 = 0
    jz0 = 0

    idec = -1
    # LL = 64 * (2 + mode_q65) # 64 * 10
    LL = 64 * 10
    NN = 63

    s_maxiters = 100

    nfft = nsps
    df = 12000.0 / nfft

    iz = int(5000.0 / df)
    txt = 85.0 * nsps / 12000.0
    jz = int((txt + 1.0) * 12000.0 / istep)

    nfa = 214.333328
    nfb = 2000.000000
    ibwa = 1
    ibwb = 11

    iseq = 0

    navg = np.zeros(2, dtype=np.int64)
    candidates_ = np.zeros((2, 20), dtype=np.float64)

    ncw = 0

    max_drift = 0
    f_max_drift = False

    s1raw_ = np.zeros((800, 7000), dtype=np.float64)

    codec = q65_init()

    # q65_dec0 df: 3,333333, nfa: 214,333328, nfb: 2000,000000

    # !Compute symbol spectra with NSTEP time-steps per symbol.
    def q65_symspec(self, iwave: npt.NDArray, iz: int, jz: int) -> npt.NDArray:
        nfft = self.nsps
        fac = (1.0 / 32767.0) * 0.01

        s1_ = np.zeros((800, 7000), dtype=np.float64)

        for j in range(0, jz, 2):  # !Compute symbol spectra at 2*step size
            i1 = j * self.istep
            i2 = i1 + self.nsps
            c0 = np.fft.fft(iwave[i1:i2] * fac, n=nfft)[:iz]  # iwave * fac ?

            s1_[j][:iz] = np.abs(c0) ** 2

            # ! For large Doppler spreads, should we smooth the spectra here? //c++   ==.EQ. !=.NE. >.GT. <.LT. >=.GE. <=.LE.
            if self.nsmo > 1:
                for _ in range(self.nsmo):
                    s1_[j] = smo121(s1_[j])

            # ! Interpolate to fill in the skipped-over spectra.
            if j >= 2:
                s1_[j - 1] = 0.5 * (s1_[j - 2] + s1_[j])

        # if lnewdat:
        #     navg[iseq]+=1
        #     ntc=fmin(navg[iseq],4) #!Averaging time constant in sequences
        #     u=1.0/ntc
        #     for (int j = 0; j < jz ; ++j)
        #         for (int z = 0; z < iz ; ++z)
        #             s1a_[iseq][j][z] = u*s1_[j][z] + (1.0-u)*s1a_[iseq][j][z]
        #     emit EmitAvgSavesQ65(navg[0],navg[1]);
        return s1_

    # void DecoderQ65::q65_ccf_22(
    # double s1_[800][7000],
    # int iz,
    # int jz,
    # double nfqso,
    # int iavg,
    # int &ipk,
    # int &jpk,
    # double &f0,
    # double &xdt,
    # bool fsdec)//,double *ccf2
    def q65_ccf_22(self, s1_: npt.NDArray[np.float64], iz: int, jz: int, nfqso: float, iavg: int, fsdec: bool):
        xdt2 = np.zeros(7000, dtype=np.float64)
        ccf3 = np.zeros(7000, dtype=np.float64)
        s1avg = np.zeros(7000, dtype=np.float64)
        # indx = np.zeros(7000, dtype=np.int64)

        mdec_df = 50
        snfa = nfqso - mdec_df
        snfb = nfqso + mdec_df
        if fsdec:
            snfa = self.nfa
            snfb = self.nfb

        self.max_drift = 0
        if self.f_max_drift:
            self.max_drift = 100.0 / self.df
        if self.max_drift > 60:
            self.max_drift = 60

        if iavg != 0:
            self.max_drift = 0

        ia = int(max(self.nfa, 100.0) / self.df)
        ib = int(min(self.nfb, 4900.0) / self.df)

        for i in range(ia, ib):
            s1avg[i] = np.sum(s1_[:jz, i])

        ccfbest = 0.0
        ibest = 0
        lagbest = 0
        idrift_best = 0
        for i in range(ia, ib):
            ccfmax_s = 0.0
            ccfmax_m = 0.0
            lagpk_s = 0
            lagpk_m = 0
            idrift_max_s = 0
            for lag in range(self.lag1, self.lag2 + 1):
                for idrift in range(-self.max_drift, self.max_drift + 1):
                    ccft = 0.0
                    for kk in range(22):
                        k = Q65_SYNC[kk] - 1
                        zz = idrift * (k - 43)
                        ii = i + (int)(zz / 85.0)
                        if ii < 0 or ii >= iz:
                            continue
                        n = self.NSTEP * k
                        j = n + lag + self.j0
                        if j > -1 and j < jz:
                            ccft += s1_[j, ii]

                    ccft -= (22.0 / jz) * s1avg[i]
                    if ccft > ccfmax_s:
                        ccfmax_s = ccft
                        lagpk_s = lag
                        idrift_max_s = idrift
                    if ccft > ccfmax_m and idrift == 0:
                        ccfmax_m = ccft
                        lagpk_m = lag

            ccf3[i] = ccfmax_m
            xdt2[i] = lagpk_m * self.dtstep

            f = i * self.df
            if ccfmax_s > ccfbest and (f >= snfa and f <= snfb):
                ccfbest = ccfmax_s
                ibest = i
                lagbest = lagpk_s
                idrift_best = idrift_max_s

        # corrp = pomAll.maxloc_da_beg_to_end(ccf3,snfa/self.df,snfb/self.df)
        corrp = np.argmax(ccf3[int(snfa / self.df):int(snfb / self.df)]) + int(snfa / self.df)
        self.xdtnd = xdt2[corrp]
        self.f0nd = nfqso + (corrp - self.i0) * self.df

        # ! Parameters for the top candidate:
        ipk = ibest - self.i0
        jpk = lagbest
        f0 = nfqso + ipk * self.df
        xdt = jpk * self.dtstep
        self.drift = self.df * idrift_best

        for i in range(ia):
            ccf3[i] = 0.0
        for i in range(ib, iz):
            ccf3[i] = 0.0

        # ! Save parameters for best candidates
        jzz = ib - ia
        if jzz < 25:
            jzz = 25
        t_s = np.zeros(7000, dtype=np.float64)
        for z in range(jzz):
            t_s[z] = ccf3[z + ia]

        # pomAll.indexx_msk(t_s, jzz - 1, indx)
        indx = np.argsort(t_s[:jzz])
        ave = pctile_shell(t_s, jzz, 50)
        base = pctile_shell(t_s, jzz, 84)

        rms = base - ave
        if rms == 0.0:
            rms = 0.000001

        ncand = 0
        maxcand = 20

        for j in range(maxcand):
            k = jzz - j - 1
            if k < 0 or k >= iz:
                continue
            i = indx[k] + ia
            f = i * self.df
            i3 = int(max(0, i - self.mode_q65))
            i4 = int(min(iz, i + self.mode_q65))
            biggest = np.max(ccf3[i3:i4])
            if ccf3[i] != biggest:
                continue
            snr = (ccf3[i] - ave) / rms
            if snr < 6.0:
                break
            self.candidates_[0][ncand] = xdt2[i]
            self.candidates_[1][ncand] = f
            ncand += 1
            if ncand > maxcand - 1:
                break  # no needed

        # ! Resort the candidates back into frequency order
        tmp_ = np.zeros((2, 25), dtype=np.float64)
        for j in range(ncand):
            tmp_[0][j] = self.candidates_[0][j]
            tmp_[1][j] = self.candidates_[1][j]
            self.candidates_[0][j] = 0.0
            self.candidates_[1][j] = 0.0
            indx[j] = 0

        if ncand > 0:
            indx = np.argsort(tmp_[1][:ncand])

        for i in range(ncand):
            self.candidates_[0][i] = tmp_[0][indx[i]]
            self.candidates_[1][i] = tmp_[1][indx[i]]

        return ipk, jpk, f0, xdt

    # void DecoderQ65::q65_ccf_85(double s1_[800][7000],int iz,int jz,double nfqso,int ia,int ia2,
    #                             int &ipk,int &jpk,double &f0,double &xdt,double &better)
    def q65_ccf_85(self, s1_: npt.NDArray[np.float64], iz: int, jz: int, nfqso: float, ia: int, ia2: int):
        # Attempt synchronization using all 85 symbols, in advance of an
        # attempt at q3 decoding.  Return ccf1 for the "red sync curve".

        ccf_ = np.zeros((310, 14000), dtype=np.float64)
        itone = np.zeros(85, dtype=np.int64)
        best = np.zeros(self.MAX_NCW, dtype=np.float64)

        # ipk = 0
        # jpk = 0
        ccf_best = 0.0
        imsg_best = -1
        for imsg in range(self.ncw):
            i = 0
            k = 0
            for j in range(85):
                if j == Q65_SYNC[i] - 1:
                    itone[j] = 0
                    i += 1
                else:
                    itone[j] = self.codewords_[imsg][k] + 1
                    k += 1

            # Compute 2D ccf using all 85 symbols in the list message
            for z in range(self.lag2 + self.ccf_offsetr):
                for j in range(self.ccf_offsetc - ia2, ia2 + self.ccf_offsetc):
                    ccf_[z][j] = 0.0

            iia = int(200.0 / self.df)
            for lag in range(self.lag1, self.lag2):
                for x in range(85):
                    j = self.j0 + self.NSTEP * x + 1 + lag
                    if j > 0 and j < jz:
                        for y in range(-ia2, ia2):
                            ii = self.i0 + self.mode_q65 * itone[x] + y
                            if ii >= iia and ii < iz:
                                ccf_[lag + self.ccf_offsetr][y + self.ccf_offsetc] += s1_[j][ii]

            ccfmax0 = 0.0
            for j in range(self.lag2 + self.ccf_offsetr):
                ccfmax = np.max(ccf_[j][self.ccf_offsetc - ia:ia + self.ccf_offsetc])
                if ccfmax > ccf_best:
                    ccf_best = ccfmax
                    ipk = np.argmax(ccf_[j][self.ccf_offsetc - ia:ia + self.ccf_offsetc]) - self.ccf_offsetc
                    jpk = j - self.ccf_offsetr
                    f0 = nfqso + ipk * self.df
                    xdt = jpk * self.dtstep
                    imsg_best = imsg

                if ccfmax > ccfmax0:
                    ccfmax0 = ccfmax

            best[imsg] = ccfmax0

        better = 0.0
        if imsg_best > -1:
            best[imsg_best] = 0.0
            tbest = np.max(best[:self.MAX_NCW])
            if tbest == 0.0:
                tbest = 0.001
            better = ccf_best / tbest

    # void DecoderQ65::q65_s1_to_s3(double s1_[800][7000],int iz,int jz,int ipk,int jpk,int LL,float *s3_1fa)
    def q65_s1_to_s3(
            self,
            s1_: npt.NDArray[np.float64],
            iz: int,
            jz: int,
            ipk: int,
            jpk: int,
            LL: int,
            s3_1fa: npt.NDArray[np.float64]
    ):
        # ! Copy synchronized symbol energies from s1 (or s1a) into s3.
        i1 = self.i0 + ipk + self.mode_q65 - 64
        i2 = i1 + LL  # int LL=64*(2+mode_q65);
        i3 = i2 - i1  # A=192 .... D=640
        if i1 > 0 and i2 < iz:
            j = self.j0 + jpk - 8
            n = 0
            for k in range(85):
                j += 8
                if self.sync[k] > 0.0:
                    continue

                if j > 0 and j < jz:
                    for i in range(i3):
                        s3_1fa[n] = s1_[j, i + i1]
                        n += 1

        q65_bzap(s3_1fa, LL)  # !Zap birdies

    # void DecoderQ65::q65_dec_q3(
    # double s1_[800][7000],
    # int iz,
    # int jz,
    # float *s3_1fa,
    # int LL,
    # int ipk,
    # int jpk,
    # double &snr2,
    # int *dat4,
    # int &idec,
    # QString &decoded)
    def q65_dec_q3(
            self,
            s1_: npt.NDArray[np.float64],
            iz: int, jz: int, LL: int, ipk: int, jpk: int
    ):
        # !Copy synchronized symbol energies from s1 into s3, then attempt a q3 decode.
        self.q65_s1_to_s3(s1_, iz, jz, ipk, jpk, LL, s3_1fa)

        if self.mode_q65 == 2:
            nsubmode = 1
        elif self.mode_q65 == 4:
            nsubmode = 2
        elif self.mode_q65 == 8:
            nsubmode = 3
        else:
            nsubmode = 0

        baud = 12000.0 / self.nsps
        for ibw in range(self.ibwa, self.ibwb):
            b90 = np.pow(1.72, ibw)
            b90ts = b90 / baud
            irc = -2
            esnodb = 0.0
            self.q65_dec1(s3_1fa, nsubmode, b90ts, esnodb, irc, dat4, decoded)
            if irc >= 0:
                snr2 = esnodb - dB(2500.0 / baud) + 3.0  # !Empirical adjustment
                idec = 3
                break

    # void DecoderQ65::q65_ap(
    # int nQSOprogresst,
    # int ipass,
    # int cont_id,
    # int cont_type,
    # bool lapcqonly,
    # int &iaptypet,
    # int *apsym0t,
    # bool *apmaskt,
    # bool *apsymbolst)
    def q65_ap(
            self,
            nQSOprogresst: int,
            ipass: int,
            cont_id: int,
            cont_type: int,
            lapcqonly: bool,

            # iaptypet
            apsym0t: npt.NDArray[np.int64],
            apmaskt: npt.NDArray[np.int64],
            apsymbolst: npt.NDArray[np.int64],
    ):
        # ! nQSOprogress
        # !   0  CALLING
        # !   1  REPLYING
        # !   2  REPORT
        # !   3  ROGER_REPORT
        # !   4  ROGERS
        # !   5  SIGNOFF
        if cont_id != self.ncontest0:
            # ! iaptype
            # !------------------------
            # !   1        CQ     ???    ???           (29+4=33 ap bits)
            # !   2        MyCall ???    ???           (29+4=33 ap bits)
            # !   3        MyCall DxCall ???           (58+4=62 ap bits)
            # !   4        MyCall DxCall RRR           (78 ap bits)
            # !   5        MyCall DxCall 73            (78 ap bits)
            # !   6        MyCall DxCall RR73          (78 ap bits)
            # bool c77[100];
            if cont_id != 0:
                i3 = 0
            n3 = 0
            # for (int i = 0; i < 78; ++i) c77[i]=0
            # TGenQ65->pack77(s_cont_cq+" LZ2HV KN23",i3,n3,c77);
        for i in range(29):
            if cont_id == 0:
                self.mcq_q65[i] = mcq_ft[i]
            else:
                self.mcq_q65[i] = c77[i]
        ncontest0 = cont_id

        for i in range(78):
            apsymbolst[i] = 0
        iaptypet = naptypes_q65[nQSOprogresst][ipass - 1]
        if lapcqonly:
            iaptypet = 1

        # Activity Type                id	type	dec-id       dec-type	dec-cq
        # "Standard"					0	0		0 = CQ		 0			0
        # "EU RSQ And Serial Number"	1	NONE	1  NONE		 NONE		NONE
        # "NA VHF Contest"				2	2		2  CQ TEST	 1			3 = CQ TEST
        # "EU VHF Contest"				3 	3		3  CQ TEST	 2			3 = CQ TEST
        # "ARRL Field Day"				4	4		4  CQ FD	 3			2 = CQ FD
        # "ARRL Inter. Digital Contest"	5	2		5  CQ TEST   1 			3 = CQ TEST
        # "WW Digi DX Contest"			6	2		6  CQ WW	 1			4 = CQ WW
        # "FT4 DX Contest"				7	2		7  CQ WW	 1			4 = CQ WW
        # "FT8 DX Contest"				8	2		8  CQ WW	 1			4 = CQ WW
        # "FT Roundup Contest"			9	5		9  CQ RU	 4			1 = CQ RU
        # "Bucuresti Digital Contest"	10 	5		10 CQ BU 	 4			5 = CQ BU
        # "FT4 SPRINT Fast Training"	11 	5		11 CQ FT 	 4			6 = CQ FT
        # "PRO DIGI Contest"			12  5		12 CQ PDC 	 4			7 = CQ PDC
        # "CQ WW VHF Contest"			13	2		13 CQ TEST	 1			3 = CQ TEST
        # "Q65 Pileup" or "Pileup"		14	2		14 CQ 		 1			0 = CQ
        # "NCCC Sprint"					15	2		15 CQ NCCC	 1			8 = CQ NCCC
        # "ARRL Inter. EME Contest"		16	6		16 CQ 		 0			0 = CQ
        # "FT Challenge"				17  6       17 CQ FTC    0          9 = CQ FTC

        # ! Conditions that cause us to bail out of AP decoding
        # !  if(ncontest.le.5 .and. iaptype.ge.3 .and. (abs(f1-nfqso).gt.napwid .and. abs(f1-nftx).gt.napwid) ) goto 900
        # !  if(ncontest.eq.6) goto 900                      !No AP for Foxes
        # !  if(ncontest.eq.7.and.f1.gt.950.0) goto 900      !Hounds use AP only below 950 Hz
        if iaptypet >= 2 and apsym0t[0] > 1:
            return
        if iaptypet >= 3 and apsym0t[29] > 1:
            return

        if iaptypet == 1:  # ! CQ or CQ RU or CQ TEST or CQ FD
            for z in range(78):
                apmaskt[z] = 0
                if z < 29:
                    apmaskt[z] = 1
                    apsymbolst[z] = self.mcq_q65[z]
            apmaskt[74] = 1
            apmaskt[75] = 1
            apmaskt[76] = 1
            apmaskt[77] = 1
            apsymbolst[74] = 0
            apsymbolst[75] = 0
            apsymbolst[76] = 1
            apsymbolst[77] = 0

        if iaptypet == 2:  # ! MyCall,???,???
            for z in range(78):
                apmaskt[z] = 0

            if cont_type == 0 or cont_type == 1:
                for z in range(29):
                    apmaskt[z] = 1
                    apsymbolst[z] = apsym0t[z]
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 1
                apsymbolst[77] = 0
            elif cont_type == 2:
                for z in range(28):
                    apmaskt[z] = 1
                    apsymbolst[z] = apsym0t[z]
                apmaskt[71] = 1
                apmaskt[72] = 1
                apmaskt[73] = 1
                apsymbolst[71] = 0
                apsymbolst[72] = 1
                apsymbolst[73] = 0
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 0
                apsymbolst[77] = 0
            elif cont_type == 3:
                for z in range(28):
                    apmaskt[z] = 1
                    apsymbolst[z] = apsym0t[z]
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 0
                apsymbolst[77] = 0
            elif cont_type == 4:
                for z in range(29):
                    apmaskt[z] = 1
                    apsymbolst[z] = apsym0t[z]
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 1
                apsymbolst[77] = 0

        if iaptypet == 3:  # ! MyCall,DxCall,???
            for z in range(78):
                apmaskt[z] = 0
            if cont_type == 0 or cont_type == 1 or cont_type == 2:
                for z in range(58):
                    apmaskt[z] = 1
                    apsymbolst[z] = apsym0t[z]
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 1
                apsymbolst[77] = 0

            elif cont_type == 3:  # then ! Field Day
                for z in range(57):
                    if z < 56:
                        apmaskt[z] = 1
                    if z < 28:
                        apsymbolst[z] = apsym0t[z]
                    if z > 28:
                        apsymbolst[z - 1] = apsym0t[z]
                apmaskt[71] = 1
                apmaskt[72] = 1
                apmaskt[73] = 1
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 0
                apsymbolst[77] = 0

            elif cont_type == 4:
                for z in range(57):
                    if z > 0:
                        apmaskt[z] = 1
                    if z < 28:
                        apsymbolst[z + 1] = apsym0t[z]
                    if z > 28:
                        apsymbolst[z] = apsym0t[z]
                apmaskt[74] = 1
                apmaskt[75] = 1
                apmaskt[76] = 1
                apmaskt[77] = 1
                apsymbolst[74] = 0
                apsymbolst[75] = 0
                apsymbolst[76] = 1
                apsymbolst[77] = 0

        if iaptypet == 4 or iaptypet == 5 or iaptypet == 6:
            for z in range(78):
                apmaskt[z] = 0

            if cont_type <= 4:
                for z in range(78):
                    apmaskt[z] = 1  # apmask(1:77)=1   //! mycall, hiscall, RRR|73|RR73
                    if z < 58:
                        apsymbolst[z] = apsym0t[z]
                apmaskt[71] = 0
                apmaskt[72] = 0
                apmaskt[73] = 0
                for z in range(19):
                    if iaptypet == 4:
                        apsymbolst[z + 58] = mrrr_ft[z]  # apsymbols(59:77)=mrrr
                    if iaptypet == 5:
                        apsymbolst[z + 58] = m73_ft[z]  # apsymbols(59:77)=m73
                    if iaptypet == 6:
                        apsymbolst[z + 58] = mrr73_ft[z]  # apsymbols(59:77)=mrr73

        return iaptypet

    # void DecoderQ65::q65_dec2(
    # float *s3_1fa,
    # int nsubmode,
    # float b90ts,
    # float &esnodb,
    # int &irc,
    # int *dat4)
    def q65_dec2(
            self,
            s3_1fa: npt.NDArray[np.float64],
            nsubmode: int,
            b90ts: float,
            dat4: npt.NDArray[np.int64],
    ):
        # ! Attempt a q0, q1, or q2 decode using spcified AP information.
        # bool c77[100];
        # float s3prob[4132] = {0.0};//row= 63 col= 64=4032
        # bool unpk77_success = false;
        #
        nFadingModel = 1
        # decoded="";

        s3prob = q65_intrinsics_ff(self.codec, s3_1fa.reshape((-1, self.NN)), nsubmode, b90ts, nFadingModel)
        irc, esnodb = q65_dec(self.codec, s3_1fa.reshape((-1, self.NN)), s3prob, self.apmask, self.apsymbols,
                              self.s_maxiters, dat4)
        print("decoded data:", dat4)

        sumd4 = 0
        for i in range(13):
            sumd4 += dat4[i]
        if sumd4 <= 0:
            irc = -2
        if irc >= 0:
            co_t = 0
            for i in range(13):
                bits = 6
                in_ = dat4[i]
                if i == 12:
                    in_ /= 2
                    bits = 5
                # SetArrayBits(in_,bits,c77,co_t)
            # decoded = TGenQ65->unpack77(c77,unpk77_success);
        return irc, esnodb

    # void DecoderQ65::q65_dec_q012(
    # float *s3_1fa,
    # double &snr2,
    # int *dat4,
    # int &idec,
    # int nQSOprogress,
    # int cont_id,
    # int cont_type)
    def q65_dec_q012(
            self,
            s3_1fa: npt.NDArray[np.float64],
            dat4: npt.NDArray[np.int64],
            nQSOprogress: int,
            cont_id: int,
            cont_type: int,
    ):
        # ! Do separate passes attempting q0, q1, q2 decodes.
        if self.mode_q65 == 2:
            nsubmode = 1
        elif self.mode_q65 == 4:
            nsubmode = 2
        elif self.mode_q65 == 8:
            nsubmode = 3
        else:
            nsubmode = 0

        baud = 12000.0 / self.nsps

        self.apsym0 = np.zeros(58, dtype=np.int64)
        self.apmask1 = np.zeros(78, dtype=np.int64)
        self.apsymbols1 = np.zeros(78, dtype=np.int64)

        lapcqonly = False
        iaptype = 0
        for ipass in range(self.npasses + 1):  # !Loop over AP passes
            self.apmask = np.zeros(13, dtype=np.int64)  # !Try first with no AP information
            self.apsymbols = np.zeros(13, dtype=np.int64)

            if ipass >= 1:
                # ! Subsequent passes use AP information appropiate for nQSOprogress
                iaptype = self.q65_ap(nQSOprogress, ipass, cont_id, cont_type, lapcqonly, self.apsym0, self.apmask1,
                                      self.apsymbols1)
                z = 0
                for i in range(13):
                    self.apmask[i] = BinToInt32(apmask1, z, z + 6)
                    self.apsymbols[i] = BinToInt32(apsymbols1, z, z + 6)
                    z += 6

            for ibw in range(self.ibwa, self.ibwb + 1):
                b90 = pow(1.72, ibw)
                b90ts = b90 / baud
                irc, esnodb = self.q65_dec2(s3_1fa, nsubmode, b90ts, dat4)
                if irc >= 0:
                    snr2 = esnodb - dB(2500.0 / baud) + 3.0  # !Empirical adjustment
                    idec = iaptype
                    return snr2, idec

    # void DecoderQ65::q65_dec0(int iavg,double *iwave,double nfqso,
    #                           bool &lclearave,bool emedelay,double &xdt,double &f0,double &snr1,
    #                           int *dat4,double &snr2,int &idec,int nQSOp,int cont_id,int cont_type,
    #                           int stageno,bool fsdec)
    def q65_dec0(
            self,
            iavg: int,
            iwave: npt.NDArray[np.float64],
            nfqso: float,
            lclearave: bool,
            emedelay: bool,
            nQSOp: int,
            cont_id: int,
            cont_type: int,
            stageno: int,
            fsdec: bool
    ) -> typing.Tuple[
        bool,
        float,
        float,
        npt.NDArray[np.int64],
        float,
        int,
    ]:
        # OUT:
        # lclearave
        # xdt
        # f0
        # --snr1
        # dat4
        # snr2
        # idec

        # Top-level routine in q65 module
        # !   - Compute symbol spectra
        # !   - Attempt sync and q3 decode using all 85 symbols
        # !   - If that fails, try sync with 22 symbols and standard q[0124] decode
        #
        # ! Input:  iavg                   0 for single-period decode, 1 for average
        # !         iwave(0:nmax-1)        Raw data
        # !         ntrperiod              T/R sequence length (s)
        # !         nfqso                  Target frequency (Hz)
        # !         ntol                   Search range around nfqso (Hz)
        # !         ndepth                 Requested decoding depth
        # !         lclearave              Flag to clear the accumulating array
        # !         emedelay               Extra delay for EME signals
        # ! Output: xdt                    Time offset from nominal (s)
        # !         f0                     Frequency of sync tone
        # !         snr1                   Relative SNR of sync signal
        # !         width                  Estimated Doppler spread
        # !         dat4(13)               Decoded message as 13 six-bit integers
        # !         snr2                   Estimated SNR of decoded signal
        # !         idec                   Flag for decing results
        # !            -1  No decode
        # !             0  No AP
        # !             1  "CQ        ?    ?"
        # !             2  "Mycall    ?    ?"
        # !             3  "MyCall HisCall ?"

        idec = -1
        LL = 64 * (2 + self.mode_q65)  # mode_q65 -- 1, 2, 3, 4
        nfft = self.nsps
        self.df = 12000.0 / nfft  # !Freq resolution = baud
        istep = self.nsps / self.NSTEP
        iz = int(5000.0 / self.df)  # !Uppermost frequency bin, at 5000 Hz
        txt = 85.0 * self.nsps / 12000.0
        jz = int((txt + 1.0) * 12000.0 / istep)  # !Number of symbol/NSTEP bins
        if self.nsps >= 6912:
            jz = int((txt + 2.0) * 12000.0 / istep)  # !For TR 60 s and higher

        ia = int(self.nfa / self.df)
        xxmax = max(10 * self.mode_q65, int(100.0 / self.df))
        ia2 = max(ia, xxmax)

        self.nsmo = max(1, int(0.5 * self.mode_q65 ** 2))

        s1_ = np.zeros((800, 7000), dtype=np.float64)
        s1a_ = np.zeros((2, jz, iz), dtype=np.float64)

        s3_1fa = np.zeros(63 * 640, dtype=np.float64)  # attention = 63*640=40320 q65d from q65_subs

        s1w_ = np.zeros((800, 7000), dtype=np.float64)
        t_s = np.zeros(700, dtype=np.float64)
        ipk = 0
        jpk = 0
        # f0a = 0.0
        # xdta = 0.0
        # smax = 0.0
        # snr1 = 0.0

        self.first = True
        if self.first:  # !Generate the sync vector
            tones_count = 85  # tones count
            self.sync = np.full(tones_count, -22.0 / 63.0, dtype=np.float64)  # !Sync tone OFF

            # tone_indices = np.setdiff1d(range(tones_count), Q65_SYNC - 1)
            # self.sync[tone_indices] = 1.0  # !Sync tone ON
            self.sync[Q65_SYNC - 1] = 1.0  # !Sync tone ON

            self.first = False

        if LL != self.LL0 or iz != self.iz0 or jz != self.jz0 or lclearave:
            self.navg[0] = 0
            self.navg[1] = 0

            self.LL0 = LL
            self.iz0 = iz
            self.jz0 = jz

            lclearave = False

        dtstep = self.nsps / (self.NSTEP * 12000.0)  # !Step size in seconds
        self.lag1 = int(-1.0 / dtstep)
        self.lag2 = int(1.0 / dtstep + 0.9999)

        if self.nsps >= 3600 and emedelay:
            self.lag2 = int(5.5 / dtstep + 0.9999)  # !Include EME

        self.j0 = int(0.5 / dtstep)
        if self.nsps >= 7200:
            self.j0 = int(1.0 / dtstep)  # !Nominal start-signal index if(nsps.ge.7200) j0=1.0/dtstep

        # s3_1fa = np.zeros(40320, dtype=np.float64)

        if iavg == 0:
            # ! Compute symbol spectra with NSTEP time bins per symbol
            s1_ = self.q65_symspec(iwave, iz, jz)
        else:
            s1_[:, :] = s1a_[self.iseq, :, :]

        self.i0 = int(nfqso / self.df)  # !Target QSO frequency
        if self.i0 - 64 < 0:
            self.i0 = 64
        if self.i0 - 64 + LL > iz - 1:
            self.i0 = iz + 64 - LL

        for j in range(jz):
            for z in range(LL):
                t_s[z] = s1_[j][z + self.i0 - 64]

            base = pctile_shell(t_s, LL, 45)
            if base == 0.0:
                base = 0.000001

            for z in range(iz):
                s1_[j][z] /= base
                self.s1raw_[j][z] = s1_[j][z]

        for j in range(jz):
            # ! Apply fast AGC to the symbol spectra
            s1max = 20.0  # !Empirical choice
            smax = np.max(s1_[j][:iz])  # smax=maxval(s1(ii1:ii2,j))
            if smax > s1max:
                s1_[j][:iz] *= s1max / smax

        dat4 = np.zeros(14, dtype=np.int64)
        if self.ncw > 0 and iavg <= 1:
            # ! Try list decoding via "Deep Likelihood".
            # ! Try to synchronize using all 85 symbols
            better = 0.0
            self.q65_ccf_85(s1_, iz, jz, nfqso, ia, ia2, ipk, jpk, f0, xdt, better)
            if better >= 1.10 or self.mode_q65 >= 8:
                self.q65_dec_q3(s1_, iz, jz, s3_1fa, LL, ipk, jpk, snr2, dat4, idec, decoded)

        # ! Get 2d CCF and ccf2 using sync symbols only
        ipk, jpk, f0a, xdta = self.q65_ccf_22(s1_, iz, jz, nfqso, iavg, fsdec)  # maybe out of bandwidth df
        if idec < 0:
            f0 = f0a
            xdt = xdta

        if idec <= 0:
            # ! The q3 decode attempt failed. Copy synchronized symbol energies from s1
            # ! into s3 and prepare to try a more general decode.
            self.q65_s1_to_s3(s1_, iz, jz, ipk, jpk, LL, s3_1fa)

        if idec < 0 and (iavg == 0 or iavg == 2):
            snr2, idec = self.q65_dec_q012(s3_1fa, dat4, nQSOp, cont_id, cont_type)

        if idec < 0 and self.max_drift != 0 and stageno == 5:
            s1w_[:, :] = s1_[:, :]

            for w3t in range(jz):
                for w3f in range(iz):
                    mm = w3f + int(self.drift * w3t / (jz * self.df))
                    if mm >= 0 and mm < iz:
                        s1w_[w3t][w3f] = s1_[w3t][mm]

            if self.ncw > 0 and iavg <= 1:  # ! Try list decoding via "Deep Likelihood".
                better = 0.0  # ! Try to synchronize using all 85 symbols
                self.q65_ccf_85(s1w_, iz, jz, nfqso, ia, ia2, ipk, jpk, f0, xdt, better)

                if better >= 1.10:
                    self.q65_dec_q3(s1w_, iz, jz, s3_1fa, LL, ipk, jpk, snr2, dat4, idec, decoded)

            if idec == 3:
                idec = 5

        return (
            lclearave,
            xdt,
            f0,
            # snr1,
            dat4,
            snr2,
            idec,
        )

    # void DecoderQ65::q65_dec1(float *s3_1fa,int nsubmode,float b90ts,float &esnodb,int &irc,
    #                           int *dat4,QString &decoded)
    def q65_dec1(
            self,
            s3_1fa: npt.NDArray[np.float64],
            nsubmode: int,
            b90ts: float,
    ):
        # ! Attmpt a full-AP list decode.
        plog = self.PLOG_MIN
        nFadingModel = 1
        q65_intrinsics_ff(self.codec, s3_1fa, nsubmode, b90ts, nFadingModel, s3prob)
        q65_dec_fullaplist(s3_1fa, s3prob, codewords_1da, ncw, esnodb, dat4, plog, irc)

        sumd4 = np.sum(dat4[:13])
        if sumd4 <= 0:
            irc = -2
        if irc >= 0 and plog > self.PLOG_MIN:
            co_t = 0
            for i in range(13):
                bits = 6
                in_ = dat4[i]
                if i == 12:
                    in_ /= 2
                    bits = 5
                # SetArrayBits(in_, bits, c77, co_t)
            # decoded = TGenQ65->unpack77(c77,unpk77_success)
        else:
            irc = -1


if __name__ == '__main__':
    with open("../data3.json") as f:
        data = json.load(f)

    iwave = np.array(data["iwave"], dtype=np.float64)

    q65 = Q65()
    q65.q65_dec0(
        iavg=0,
        iwave=iwave,
        nfqso=1000,
        lclearave=False,
        emedelay=False,
        nQSOp=0,
        cont_id=0,
        cont_type=0,
        stageno=0,
        fsdec=False,
    )
