// sim_mdm_server.cpp
// Hardware-independent version of the Asgard Multi DM server
//  g++ -O2 -std=c++17 -o sim_mdm_server sim_mdm_server.cpp -lImageStreamIO -lpthread

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <iostream>
#include <ImageStreamIO.h>
#include <algorithm>
//#include <commander/commander.h>

#define LINESIZE 256
#define CMDSIZE 200

int wxsz, wysz;
int ii;
IMAGE **shmarray = NULL;
int nch = 5;
int dms = 12;
int nact = 140;
int nvact = 144;
int keepgoing = 0;
int nch_prev = 0;
char dashline[80] = "-----------------------------------------------------------------------------\n";

int ndm = 4;
int simmode = 1;
int timelog = 0;
char drv_status[8] = "idle";

pthread_t tid_loop;
unsigned int targs[4] = {1, 2, 3, 4};

int shm_setup();
void* dm_control_loop(void *_dmid);
double* map2D_2_cmd(double *map2D);

int shm_setup() {
    int ii, kk;
    int shared = 1;
    int NBkw = 10;
    long naxis = 2;
    uint8_t atype = _DATATYPE_DOUBLE;
    uint32_t *imsize;
    char shmname[20];

    imsize = (uint32_t *) malloc(sizeof(uint32_t) * naxis);
    imsize[0] = dms;
    imsize[1] = dms;

    if (shmarray != NULL) {
        for (kk = 0; kk < ndm; kk++)
            for (ii = 0; ii < nch_prev; ii++)
                ImageStreamIO_destroyIm(&shmarray[kk][ii]);
        free(shmarray);
        shmarray = NULL;
    }

    shmarray = (IMAGE**) malloc(ndm * sizeof(IMAGE*));
    for (kk = 0; kk < ndm; kk++) {
        shmarray[kk] = (IMAGE*) malloc((nch + 1) * sizeof(IMAGE));
    }

    for (kk = 0; kk < ndm; kk++) {
        for (ii = 0; ii < nch; ii++) {
            sprintf(shmname, "dm%ddisp%02d", kk + 1, ii);
            ImageStreamIO_createIm_gpu(&shmarray[kk][ii], shmname, naxis, imsize, atype, -1,
                                       shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);
        }
        sprintf(shmname, "dm%d", kk + 1);
        ImageStreamIO_createIm_gpu(&shmarray[kk][nch], shmname, naxis, imsize, atype, -1,
                                   shared, IMAGE_NB_SEMAPHORE, NBkw, MATH_DATA);
    }
    free(imsize);
    return 0;
}

double* map2D_2_cmd(double *map2D) {
    int ii, jj = 0;
    double* cmd = (double*) malloc(nact * sizeof(double));
    for (ii = 0; ii < nvact; ii++) {
        if ((ii == 0) || (ii == 11) || (ii == 132) || (ii == 143)) continue;
        cmd[jj++] = map2D[ii];
    }
    return cmd;
}

void* dm_control_loop(void *_dmid) {
    uint64_t cntrs[nch];
    int ii, kk;
    double tmp_map[nvact];
    struct timespec now;

    unsigned int dmid = *((unsigned int *) _dmid);
    FILE* fd = nullptr;
    char fname[20];
    sprintf(fname, "speed_log_%1d.log", dmid);
    if (timelog) fd = fopen(fname, "w");

    for (ii = 0; ii < nch; ii++)
        cntrs[ii] = shmarray[dmid - 1][nch].md->cnt0;

    while (keepgoing > 0) {
        ImageStreamIO_semwait(&shmarray[dmid - 1][nch], 1);
        for (ii = 0; ii < nch; ii++)
            cntrs[ii] = shmarray[dmid - 1][ii].md->cnt0;

        for (ii = 0; ii < nvact; ii++) {
            tmp_map[ii] = 0.0;
            for (kk = 0; kk < nch; kk++)
                tmp_map[ii] += shmarray[dmid - 1][kk].array.D[ii];
            tmp_map[ii] = std::clamp(tmp_map[ii], 0.0, 1.0);
        }

        shmarray[dmid - 1][nch].md->write = 1;
        for (ii = 0; ii < nvact; ii++)
            shmarray[dmid - 1][nch].array.D[ii] = tmp_map[ii];
        shmarray[dmid - 1][nch].md->cnt1 = 0;
        shmarray[dmid - 1][nch].md->cnt0++;
        shmarray[dmid - 1][nch].md->write = 0;

        clock_gettime(CLOCK_REALTIME, &now);
        if (timelog) fprintf(fd, "%f\n", 1.0 * now.tv_sec + 1e-9 * now.tv_nsec);
    }
    if (timelog && fd) fclose(fd);
    return nullptr;
}

void start() {
    if (keepgoing == 0) {
        keepgoing = 1;
        std::cout << "DM control loop START\n";
        for (int kk = 0; kk < ndm; kk++)
            pthread_create(&tid_loop, NULL, dm_control_loop, &targs[kk]);
        snprintf(drv_status, sizeof(drv_status), "running");
    } else {
        std::cout << "DM control loop already running!\n";
    }
}

void stop() {
    keepgoing = 0;
    snprintf(drv_status, sizeof(drv_status), "idle");
}

std::string status() {
    return drv_status;
}

int get_nch() {
    return nch;
}

void set_nch(int ival) {
    nch_prev = nch;
    nch = ival;
    shm_setup();
    std::cout << "Success: # channels = " << ival << "\n";
}

void reset(int dmid, int channel) {
    double reset_map[nvact] = {0};
    double *live_channel;

    if (dmid > ndm || dmid <= 0) return;

    if (channel < 0) {
        for (int kk = 0; kk < nch; kk++) {
            live_channel = shmarray[dmid - 1][kk].array.D;
            shmarray[dmid - 1][kk].md->write = 1;
            memcpy(live_channel, reset_map, sizeof(double) * nvact);
            shmarray[dmid - 1][kk].md->cnt0++;
            ImageStreamIO_sempost(&shmarray[dmid - 1][kk], -1);
            shmarray[dmid - 1][kk].md->write = 0;
        }
    } else if (channel < nch) {
        live_channel = shmarray[dmid - 1][channel].array.D;
        shmarray[dmid - 1][channel].md->write = 1;
        memcpy(live_channel, reset_map, sizeof(double) * nvact);
        shmarray[dmid - 1][channel].md->cnt0++;
        ImageStreamIO_sempost(&shmarray[dmid - 1][channel], -1);
        shmarray[dmid - 1][channel].md->write = 0;
    }
}

void quit() {
    if (keepgoing == 1) stop();
    std::cout << "DM driver server shutting down!\n";
    if (shmarray != nullptr) {
        for (int kk = 0; kk < ndm; kk++) {
            for (int ii = 0; ii < nch + 1; ii++)
                ImageStreamIO_destroyIm(&shmarray[kk][ii]);
        }
        free(shmarray);
        shmarray = nullptr;
    }
    exit(0);
}

// namespace co = commander;
// COMMANDER_REGISTER(m) {
//     using namespace co::literals;
//     m.def("start", start, "Start DM loop");
//     m.def("stop", stop, "Stop DM loop");
//     m.def("status", status, "Return DM server status");
//     m.def("get_nch", get_nch, "Get number of channels");
//     m.def("set_nch", set_nch, "Set number of channels");
//     m.def("reset", reset, "Reset channel(s) of a DM");
//     m.def("quit", quit, "Quit server");
// }

int main(int argc, char** argv) {
    std::cout << dashline;
    std::cout << "Simulated DM scenario: no drivers connected\n";

    shm_setup();
    start();

    // Wait for user input to quit
    std::cout << "Press Enter to stop...\n";
    std::cin.get();

    stop();
    quit();
    return 0;
}
// int main(int argc, char** argv) {
//     std::cout << dashline;
//     std::cout << "Simulated DM scenario: no drivers connected\n";
//     shm_setup();
//     start();
//     co::Server s(argc, argv);
//     s.run();
//     quit();
//     return 0;
// }
