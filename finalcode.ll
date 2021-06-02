; ModuleID = "D:\CompilatorG\compiler\codegen.py"
target triple = "i686-pc-windows-msvc"
target datalayout = ""

define void @"global"() 
{
entry:
  ret void
}

declare i32 @"printf"(i32* %".1", ...) 

@"fstr" = internal constant [5 x i8] c"%i \0a\00"
define i32 @"sum"(i32 %".1", i32 %".2") 
{
entry:
  %"res" = add i32 %".1", %".2"
  ret i32 %"res"
}

define float @"fsum"(float %".1", float %".2") 
{
entry:
  %"res" = fadd float %".1", %".2"
  ret float %"res"
}

define i32 @"sub"(i32 %".1", i32 %".2") 
{
entry:
  %"res" = sub i32 %".1", %".2"
  ret i32 %"res"
}

define float @"fsub"(float %".1", float %".2") 
{
entry:
  %"res" = fsub float %".1", %".2"
  ret float %"res"
}
