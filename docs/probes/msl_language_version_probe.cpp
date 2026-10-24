#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstdio>
int main(){ auto* dev=MTL::CreateSystemDefaultDevice();
 for (int v : {262144, 262145, 262146, 262147, 327680, 327681}) {
  auto* o=MTL::CompileOptions::alloc()->init(); o->setLanguageVersion((MTL::LanguageVersion)v); NS::Error* e=nullptr;
  auto* l=dev->newLibrary(NS::String::string("#include <metal_stdlib>\nkernel void k(){}\n",NS::UTF8StringEncoding),o,&e);
  printf("languageVersion %d (%d.%d): %s\n", v, v>>16, v&0xffff, l?"OK":(e?e->localizedDescription()->utf8String():"fail")); }
}
