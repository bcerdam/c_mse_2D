#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

double** read_csv(const char* filename) {
    double** data = NULL;
    FILE* file = fopen(filename, "r");
    char line[1024];
    int row_count = 0;
    int col_count = 0;
    
    if (file == NULL) {
        printf("Could not open file: %s\n", filename);
        return NULL;
    }
    
    // Count the number of rows and columns in the file
    while (fgets(line, 1024, file)) {
        col_count = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            col_count++;
            token = strtok(NULL, ",");
        }
        row_count++;
    }
    
    // Allocate memory for the data array
    data = (double**)malloc(row_count * sizeof(double*));
    for (int i = 0; i < row_count; i++) {
        data[i] = (double*)malloc(col_count * sizeof(double));
    }
    
    // Reset the file pointer and read the data into the array
    fseek(file, 0, SEEK_SET);
    int row = 0;
    while (fgets(line, 1024, file)) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            data[row][col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    
    fclose(file);
    return data;
}


double** coarse_graining(double** arr, int rows, int cols, int scale_factor) {
    if (scale_factor == 1) {
        return arr;
    }
    int new_rows = rows / scale_factor;
    int new_cols = cols / scale_factor;
    double** cg_arr = malloc(new_rows * sizeof(double*));
    for (int i = 0; i < new_rows; i++) {
        cg_arr[i] = malloc(new_cols * sizeof(double));
        for (int j = 0; j < new_cols; j++) {
            double sum = 0;
            for (int x = i * scale_factor; x < (i + 1) * scale_factor; x++) {
                for (int y = j * scale_factor; y < (j + 1) * scale_factor; y++) {
                    sum += arr[x][y];
                }
            }
            cg_arr[i][j] = sum / (scale_factor * scale_factor);
        }
    }
    return cg_arr;
}


double max_distance(int m, double **image, int i, int j, int a, int b, int distance_type) {
    if (distance_type == 0){
        double max_dist = 0;
        for(int k=0; k<m; k++) {
            for(int l=0; l<m; l++) {
                double dist = fabs(image[i+k][j+l] - image[a+k][b+l]);
                if(dist > max_dist) {
                    max_dist = dist;
                }
            }
        }
        return max_dist;
    }
    else{
        double sum_dist = 0;
        for(int k=0; k<m; k++) {
            for(int l=0; l<m; l++) {
                double dist = powf(fabs(image[i+k][j+l] - image[a+k][b+l]), distance_type);
                sum_dist += dist;
            }
        }
        sum_dist = powf(sum_dist, (1.0/distance_type));
        return sum_dist;
    }
}

double fuzzy_membership(double distance, double r, double delta) {
    if (r == 0){
        return 1;
    }
    else{
        return expf(powf(distance, 2) * log(delta) / powf(r, 2.0));
    }
}

float calculate_U_ij_m(double **image, int i, int j, int m, double r, int H, int W, double delta, int fuzzy, int distance_type) {
    double count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            double dist = max_distance(m, image, i, j, a, b, distance_type);
            if (a == i && b == j) {
                continue;
            }
            else{
                if (fuzzy == 0){
                    if (dist <= r){
                        count ++;
                    }
                }
                else if (fuzzy == 1){
                    count += fuzzy_membership(dist, r, delta);
                }
            }
        }
    }
    return (float) count / (N_m-1);
}

float calculate_U_ij_m_plus_one(double **image, int i, int j, int m, double r, int H, int W, double delta, int fuzzy, int distance_type) {
    double count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            double dist = max_distance(m+1, image, i, j, a, b, distance_type);
            if (a == i && b == j) {
                continue;
            }
            else{
                if (fuzzy == 0){
                    if (dist <= r){
                        count ++;
                    }
                }
                else if (fuzzy == 1){
                    count += fuzzy_membership(dist, r, delta);
                }
            }
        }
    }
    return (float) count / (N_m-1);
}


float calculate_U_m(double **image, int m, double r, int H, int W, double delta, int fuzzy, int distance_type) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(32)
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += calculate_U_ij_m(image, i, j, m, r, H, W, delta, fuzzy, distance_type);
        }
    }
    #pragma omp barrier
    float average;
    #pragma omp critical
    {
        average = sum / ((H - m) * (W - m));
    }
    return average;
    // return sum / ((H - m) * (W - m));
}


float calculate_U_m_plus_one(double **image, int m, double r, int H, int W, double delta, int fuzzy, int distance_type) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(32)
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += calculate_U_ij_m_plus_one(image, i, j, m, r, H, W, delta, fuzzy, distance_type);
        }
    }
    #pragma omp barrier
    float average;
    #pragma omp critical
    {
        average = sum / ((H - m) * (W - m));
    }
    return average;

    // return sum / ((H - m) * (W - m));
}


float negative_logarithm(float um, float umplus1) {
    if (um == 0) {
        return 0;
    }
    else {  
        float result = -log(umplus1 / um);
        return result;  
    }
}


int main(int argc, char *argv[]) {
    
    char* csv_path = argv[1];
    double** data = read_csv(csv_path);
    int scales = atoi(argv[2]);
    int rows = atoi(argv[3]);
    int cols = atoi(argv[4]);
    int m = atoi(argv[5]);
    double r = atof(argv[6]);
    double delta = atof(argv[7]);
    int fuzzy = atoi(argv[8]);
    int distance_type = atoi(argv[9]);


    double* n_values = malloc(scales * sizeof(double));

    for (int i = 1; i <= scales; i++) {
        double** coarse_data = coarse_graining(data, rows, cols, i);
        float U_m = calculate_U_m(coarse_data, m, r, rows/i, cols/i, delta, fuzzy, distance_type);
        float U_m_plus_one = calculate_U_m_plus_one(coarse_data, m, r, rows/i, cols/i, delta, fuzzy, distance_type);
        float n = negative_logarithm(U_m, U_m_plus_one);
        n_values[i-1] = n;
    }

    
    

    for (int i = 0; i < scales; i++) {
        printf("%f ", n_values[i]);
    }
    printf("\n");


    return 0;
}


/*
gcc mse_2D.c -o mse_2D -lm
./mse_2D
*/

//  Para imágenes RGB

double** rgb_read_csv(const char* filename1, const char* filename2, const char* filename3) {
    double** data = NULL;
    FILE* file1 = fopen(filename1, "r");
    FILE* file2 = fopen(filename2, "r");
    FILE* file3 = fopen(filename3, "r");
    char line[1024];
    int row_count = 0;
    int col_count = 0;
    
    if (file1 == NULL || file2 == NULL || file3 == NULL) {
        printf("Could not open one or more files.\n");
        return NULL;
    }
    
    // Count the number of rows and columns in the files
    while (fgets(line, 1024, file1)) {
        col_count = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            col_count++;
            token = strtok(NULL, ",");
        }
        row_count++;
    }
    
    // Allocate memory for the data array
    data = (double**)malloc(3 * sizeof(double*));
    for (int i = 0; i < 3; i++) {
        data[i] = (double*)malloc(row_count * col_count * sizeof(double));
    }
    
    // Reset the file pointers and read the data into the array
    fseek(file1, 0, SEEK_SET);
    fseek(file2, 0, SEEK_SET);
    fseek(file3, 0, SEEK_SET);
    int row = 0;
    while (fgets(line, 1024, file1)) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            data[0][row * col_count + col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    
    row = 0;
    while (fgets(line, 1024, file2)) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            data[1][row * col_count + col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    
    row = 0;
    while (fgets(line, 1024, file3)) {
        int col = 0;
        char* token = strtok(line, ",");
        while (token != NULL) {
            data[2][row * col_count + col] = atof(token);
            col++;
            token = strtok(NULL, ",");
        }
        row++;
    }
    
    fclose(file1);
    fclose(file2);
    fclose(file3);
    return data;
}


float rgb_calculate_U_ij_m(double** image_r, double** image_g, double** image_b, int i, int j, int m, float* r, int H, int W, double delta, int fuzzy, int distance_type) {
    float count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            float dist_r = max_distance(m, image_r, i, j, a, b, distance_type);
            float dist_g = max_distance(m, image_g, i, j, a, b, distance_type);
            float dist_b = max_distance(m, image_b, i, j, a, b, distance_type);
            if (a == i && b == j) {
                continue;
            }
            else {
                if (fuzzy == 0) {
                    if (dist_r <= r[0] && dist_g <= r[1] && dist_b <= r[2]) {
                        count++;
                    }
                }
                else if (fuzzy == 1) {
                    count += fuzzy_membership(dist_r, r[0], delta) * fuzzy_membership(dist_g, r[1], delta) * fuzzy_membership(dist_b, r[2], delta);
                }
            }
        }
    }
    return count / (N_m - 1);
}

float rgb_calculate_U_ij_m_plus_one(double** image_r, double** image_g, double** image_b, int i, int j, int m, float* r, int H, int W, double delta, int fuzzy, int distance_type) {
    float count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            float dist_r = max_distance(m + 1, image_r, i, j, a, b, distance_type);
            float dist_g = max_distance(m + 1, image_g, i, j, a, b, distance_type);
            float dist_b = max_distance(m + 1, image_b, i, j, a, b, distance_type);
            if (a == i && b == j) {
                continue;
            }
            else {
                if (fuzzy == 0) {
                    if (dist_r <= r[0] && dist_g <= r[1] && dist_b <= r[2]) {
                        count++;
                    }
                }
                else if (fuzzy == 1) {
                    count += fuzzy_membership(dist_r, r[0], delta) * fuzzy_membership(dist_g, r[1], delta) * fuzzy_membership(dist_b, r[2], delta);
                }
            }
        }
    }
    return count / (N_m - 1);
}

float rgb_calculate_U_m(double** image_r, double** image_g, double** image_b, int m, float* r, int H, int W, double delta, int fuzzy, int distance_type) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(32)
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += rgb_calculate_U_ij_m(image_r, image_g, image_b, i, j, m, r, H, W, delta, fuzzy, distance_type);
        }
    }
    #pragma omp barrier
    float average;
    #pragma omp critical
    {
        average = sum / ((H - m) * (W - m));
    }
    return average;
}

float rgb_calculate_U_m_plus_one(double** image_r, double** image_g, double** image_b, int m, float* r, int H, int W, double delta, int fuzzy, int distance_type) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) num_threads(32)
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += rgb_calculate_U_ij_m_plus_one(image_r, image_g, image_b, i, j, m, r, H, W, delta, fuzzy, distance_type);
        }
    }
    #pragma omp barrier
    float average;
    #pragma omp critical
    {
        average = sum / ((H - m) * (W - m));
    }
    return average;
}

float rgb_negative_logarithm(float um, float umplus1) {
    if (um == 0) {
        return 0;
    }
    else {  
        float result = -log(umplus1 / um);
        return result;  
    }
}

// int main(int argc, char* argv[]) {
//     // if (argc < 14) {
//     //     printf("Usage: %s csv_path1 csv_path2 csv_path3 scales rows cols m r1 r2 r3 delta fuzzy distance_type\n", argv[0]);
//     //     return 1;
//     // }

//     const char* csv_path1 = argv[1];
//     const char* csv_path2 = argv[2];
//     const char* csv_path3 = argv[3];

//     double** data = rgb_read_csv(csv_path1, csv_path2, csv_path3);

//     int scales = atoi(argv[4]);
//     int rows = atoi(argv[5]);
//     int cols = atoi(argv[6]);
//     int m = atoi(argv[7]);

//     float r[3] = {atof(argv[8]), atof(argv[9]), atof(argv[10])};
//     double delta = atof(argv[11]);
//     int fuzzy = atoi(argv[12]);
//     int distance_type = atoi(argv[13]);

//     double* n_values = malloc(scales * sizeof(double));

//     for (int i = 1; i <= scales; i++) {
//         double** coarse_data_r = coarse_graining(data, rows, cols, i);
//         double** coarse_data_g = coarse_graining(data, rows, cols, i);
//         double** coarse_data_b = coarse_graining(data, rows, cols, i);

//         double U_m = rgb_calculate_U_m(coarse_data_r, coarse_data_g, coarse_data_b, m, r, rows / i, cols / i, delta, fuzzy, distance_type);
//         double U_m_plus_one = rgb_calculate_U_m_plus_one(coarse_data_r, coarse_data_g, coarse_data_b, m, r, rows / i, cols / i, delta, fuzzy, distance_type);
//         double n = rgb_negative_logarithm(U_m, U_m_plus_one);
//         n_values[i - 1] = n;
//     }

//     for (int i = 0; i < scales; i++) {
//         printf("%f ", n_values[i]);
//     }
//     printf("\n");

//     free(n_values);

//     return 0;
// }