## Тип массива
# Make
make double - double массив
make float - float массив
# Cmake
cmake -DUSE_DOUBLE=ON .. - double массив
cmake -DUSE_DOUBLE=OFF .. float массив

## Результаты
make double - Sum: 4.89582e-11
make float - Sum: -0.0277862

cmake double - Sum: 6.27585e-10
cmake float - Sum: -0.0277862
