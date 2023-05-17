#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

double fuzzy_membership(double distance, double r) {
    if (r == 0){
        return 1;
    }
    else{
        return expf(-powf(distance, 2.0) / powf(r, 2.0));
    }
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


double max_distance(int m, double **image, int i, int j, int a, int b) {
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

float calculate_U_ij_m(double **image, int i, int j, int m, double r, int H, int W) {
    double count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            // double dist = max_distance(m, image, i, j, a, b);
            if (a == i && b == j) {
                continue;
            }
            if (max_distance(m, image, i, j, a, b) <= r) {
                // printf("%f %f \n", max_distance(m, image, i, j, a, b), r);
                count++;
                // count += fuzzy_membership(dist, r);
            }
        }
    }
    // printf("Um %d: \n", count / (N_m-1));
    return (float) count / (N_m-1);
}

float calculate_U_ij_m_plus_one(double **image, int i, int j, int m, double r, int H, int W) {
    double count = 0;
    int N_m = (H - m) * (W - m);
    for (int a = 0; a < H - m; a++) {
        for (int b = 0; b < W - m; b++) {
            // double dist = max_distance(m, image, i, j, a, b);
            if (a == i && b == j) {
                continue;
            }
            if (max_distance(m+1, image, i, j, a, b) <= r) {
                count++;
                // count += fuzzy_membership(dist, r);
            }
        }
    }
    return (float) count / (N_m-1);
}


float calculate_U_m(double **image, int m, double r, int H, int W) {
    float sum = 0.0;
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += calculate_U_ij_m(image, i, j, m, r, H, W);
        }
    }
    // printf("Um %f: \n", sum / ((H - m) * (W - m)));
    return sum / ((H - m) * (W - m));
}


float calculate_U_m_plus_one(double **image, int m, double r, int H, int W) {
    float sum = 0.0;
    for (int i = 0; i < H - m; i++) {
        for (int j = 0; j < W - m; j++) {
            sum += calculate_U_ij_m_plus_one(image, i, j, m, r, H, W);
        }
    }
    // printf("Ump %f: \n", sum / ((H - m) * (W - m)));
    return sum / ((H - m) * (W - m));
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
    double* n_values = malloc(scales * sizeof(double));

    for (int i = 1; i <= scales; i++) {
        double** coarse_data = coarse_graining(data, rows, cols, i);
        float U_m = calculate_U_m(coarse_data, m, r, rows/i, cols/i);
        // printf("%f \n", U_m);
        float U_m_plus_one = calculate_U_m_plus_one(coarse_data, m, r, rows/i, cols/i);
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