%module ubench

%include "std_vector.i"

%template(IntVector) std::vector<int>;

%{
#include "tf_cpu.hpp"
%}

%{ // This declaration is for swig. %}
%include "tf_cpu.hpp"

%pythoncode %{
# pylint: skip-file
%}
