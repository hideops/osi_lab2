#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
int ready = 0, step = 0;
double **A = NULL, *X = NULL;
int N = 0, T = 0;

typedef struct { 
    int id;
    int from;
    int to; 
} thr_t;

long long now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

double** alloc(int n) {
    double **a = malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) a[i] = malloc((n+1) * sizeof(double));
    return a;
}

void free_mat(double **a, int n) {
    for (int i = 0; i < n; i++) free(a[i]);
    free(a);
}

void rand_mat(double **a, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = (rand() % 1000) / 100.0;
            if (i == j) a[i][j] += 50.0;
        }
        a[i][n] = (rand() % 1000) / 100.0;
    }
}

void wait_all() {
    pthread_mutex_lock(&mtx);
    ready++;
    if (ready == T) {
        ready = 0;
        step++;
    } else {
        int cur = step;
        while (cur == step) pthread_mutex_unlock(&mtx), pthread_mutex_lock(&mtx);
    }
    pthread_mutex_unlock(&mtx);
}

void seq() {
    double **a = alloc(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= N; j++)
            a[i][j] = A[i][j];
    
    for (int k = 0; k < N; k++) {
        int mr = k;
        for (int i = k+1; i < N; i++)
            if (fabs(a[i][k]) > fabs(a[mr][k])) mr = i;
        
        if (mr != k)
            for (int j = k; j <= N; j++) {
                double t = a[k][j];
                a[k][j] = a[mr][j];
                a[mr][j] = t;
            }
        
        for (int j = k; j <= N; j++) a[k][j] /= a[k][k];
        
        for (int i = k+1; i < N; i++) {
            double f = a[i][k];
            for (int j = k; j <= N; j++) a[i][j] -= f * a[k][j];
        }
    }
    
    X = malloc(N * sizeof(double));
    for (int i = N-1; i >= 0; i--) {
        X[i] = a[i][N];
        for (int j = i+1; j < N; j++) X[i] -= a[i][j] * X[j];
        X[i] /= a[i][i];
    }
    
    free_mat(a, N);
}

void* par(void* arg) {
    thr_t *d = (thr_t*)arg;
    
    for (int k = 0; k < N-1; k++) {
        if (d->id == 0) {
            pthread_mutex_lock(&mtx);
            int mr = k;
            for (int i = k+1; i < N; i++)
                if (fabs(A[i][k]) > fabs(A[mr][k])) mr = i;
            
            if (mr != k)
                for (int j = k; j <= N; j++) {
                    double t = A[k][j];
                    A[k][j] = A[mr][j];
                    A[mr][j] = t;
                }
            pthread_mutex_unlock(&mtx);
        }
        
        wait_all();
        
        if (d->id == 0) {
            double div = A[k][k];
            for (int j = k; j <= N; j++) A[k][j] /= div;
        }
        
        wait_all();
        
        for (int i = d->from; i < d->to; i++) {
            if (i > k) {
                double f = A[i][k];
                for (int j = k; j <= N; j++) A[i][j] -= f * A[k][j];
            }
        }
        
        wait_all();
    }
    
    wait_all();
    
    if (d->id == 0) {
        X = malloc(N * sizeof(double));
        for (int i = N-1; i >= 0; i--) {
            X[i] = A[i][N];
            for (int j = i+1; j < N; j++) X[i] -= A[i][j] * X[j];
            X[i] /= A[i][i];
        }
    }
    
    wait_all();
    
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        write(STDOUT_FILENO, "usage ./program size threads\n", 
              strlen("usage ./program size threads\n"));
        write(STDOUT_FILENO, "example ./program 500 4\n", 
              strlen("example ./program 500 4\n"));
        return 1;
    }
    
    N = atoi(argv[1]);
    T = atoi(argv[2]);
    
    if (N <= 0 || T <= 0) return 1;
    if (T > N) T = N;
    
    char info[80];
    int len = snprintf(info, sizeof(info), "size %d, threads %d, pid %d\n", N, T, getpid());
    write(STDOUT_FILENO, info, len);
    
    A = alloc(N);
    rand_mat(A, N);

    write(STDOUT_FILENO, "sequential \n", strlen("sequential \n"));
    
    long long t1 = now();
    seq();
    long long ts = now() - t1;
    snprintf(info, sizeof(info), "time  %lld microseconds\n", ts);
    write(STDOUT_FILENO, info, strlen(info));
    
    free(X);
    X = NULL;
    rand_mat(A, N);
    ready = step = 0;
    
    write(STDOUT_FILENO, "parallel \n", strlen("parallel \n"));
    
    pthread_t thr[T];
    thr_t data[T];
    int per = N / T, ext = N % T, cur = 0;
    
    for (int i = 0; i < T; i++) {
        data[i].id = i;
        data[i].from = cur;
        data[i].to = cur + per + (i < ext ? 1 : 0);
        cur = data[i].to;
    }
    
    t1 = now();
    for (int i = 0; i < T; i++) pthread_create(&thr[i], NULL, par, &data[i]);
    for (int i = 0; i < T; i++) pthread_join(thr[i], NULL);
    long long tp = now() - t1;
    
    snprintf(info, sizeof(info), "time %lld microseconds\n", tp);
    write(STDOUT_FILENO, info, strlen(info));
    
    double sp = (double)ts / tp;
    double ef = (sp / T) * 100.0;
    write(STDOUT_FILENO, "results \n", strlen("results \n"));
    snprintf(info, sizeof(info), "speedup  %.3f\nefficiency %.1f%%\n", sp, ef);
    write(STDOUT_FILENO, info, strlen(info));
    write(STDOUT_FILENO, "first 3 solutions \n", strlen("first 3 solutions \n"));
    for (int i = 0; i < 3 && i < N; i++) {
        snprintf(info, sizeof(info), "x[%d] = %.6f\n", i, X[i]);
        write(STDOUT_FILENO, info, strlen(info));
    }
    free_mat(A, N);
    free(X);
    write(STDOUT_FILENO, info, strlen(info));
    return 0;
}