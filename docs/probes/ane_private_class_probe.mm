#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <cstdio>
int main() { setvbuf(stdout, NULL, _IONBF, 0);
  printf("step1 dlopen\n");
  void* h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
  printf("dlopen: %s\n", h ? "OK" : dlerror());
  if (!h) { h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/Versions/A/AppleNeuralEngine", RTLD_NOW); printf("dlopen2: %s\n", h ? "OK" : dlerror()); }
  const char* names[] = {"_ANEInMemoryModel","_ANEInMemoryModelDescriptor","_ANEClient","_ANEModel","_ANERequest","_ANEIOSurfaceObject","_ANEDeviceInfo","_ANECompiler","ANEServicesLog","_ANEProgramForEvaluation","_ANEBuffer","_ANEQoSMapper"};
  for (auto n : names) { Class c = objc_getClass(n); printf("  %-30s %s\n", n, c ? "present" : "absent"); }
  Class di = objc_getClass("_ANEDeviceInfo");
  if (di) { const char* sels[] = {"hasANE","numANEs","numANECores","aneArchitectureType","aneSubType","aneBoardType","productName","isInternalBuild"};
    for (auto s : sels) { SEL sel = sel_registerName(s); if (class_getClassMethod(di, sel)) { const char* enc = method_getTypeEncoding(class_getClassMethod(di, sel)); printf("  _ANEDeviceInfo +%s (enc %s)", s, enc); 
        if (enc[0]=='B') printf(" = %d", (int)((BOOL(*)(Class,SEL))objc_msgSend)(di, sel)); else if (enc[0]=='I'||enc[0]=='i'||enc[0]=='Q'||enc[0]=='q') printf(" = %lld", (long long)((long long(*)(Class,SEL))objc_msgSend)(di, sel)); else if (enc[0]=='@') { id r = ((id(*)(Class,SEL))objc_msgSend)(di, sel); printf(" = %s", r ? [[r description] UTF8String] : "nil"); } printf("\n"); } } }
  return 0; }
