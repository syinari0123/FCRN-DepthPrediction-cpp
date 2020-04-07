/**
* MIT License
* 
* Copyright (c) 2019 Sergio Izquierdo
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef CPPFLOW_MODEL_H
#define CPPFLOW_MODEL_H

#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <tensorflow/c/c_api.h>
#include "Tensor.h"

class Tensor;

class Model {
public:
    explicit Model(const std::string&);

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Model(const Model &model) = delete;
    Model(Model &&model) = default;
    Model& operator=(const Model &model) = delete;
    Model& operator=(Model &&model) = default;

    ~Model();

    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
    std::vector<std::string> get_operations() const;

    // Original Run
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

private:
    TF_Graph* graph;
    TF_Session* session;
    TF_Status* status;

    // Read a file from a string
    static TF_Buffer* read(const std::string&);

    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;

public:
    friend class Tensor;
};


#endif //CPPFLOW_MODEL_H
