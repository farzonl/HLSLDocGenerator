import difflib
import re
import sys

# Function to preprocess the shader file and remove
# parts of the diff we know will always be different
def remove_know_shader_differences(shader_text):
    # shader hash will be different between shader models so delete it
    shader_text = re.sub(r'shader hash: .*\n', '', shader_text)
    
    # Remove lines containing !1 = or !3 = (ignore these metadata)
    # !1 is the shader model  minor number
    # !3 is the shader stage, major number and minor number
    shader_text = re.sub(r'!1 = !{.*\n', '', shader_text)
    shader_text = re.sub(r'!3 = !{.*\n', '', shader_text)
    shader_text = re.sub(r'!9 = !{.*\n', '', shader_text)
        
    return shader_text

# Function to diff two shader files' function bodies
def diff_function_bodies(shader1, shader2):
    # Preprocess both shader files to remove irrelevant parts
    shader1_clean = remove_know_shader_differences(shader1)
    shader2_clean = remove_know_shader_differences(shader2)

    diff = difflib.unified_diff(
        shader1_clean.splitlines(), shader2_clean.splitlines(),
        fromfile='Shader 1', tofile='Shader 2', lineterm=''
    )
    
    return '\n'.join(diff)

# Example shader file contents (replace with actual file contents)
shader1 = """;
; Note: shader requires additional functionality:
;       Wave level operations
;
; shader hash: aa0dfa1c19e5ed82ecd1c79e54b713d4
;
; Buffer Definitions:
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

@dx.nothing.a = internal constant [1 x i32] zeroinitializer

; Function Attrs: nounwind
define <4 x i32> @"\01?fn@@YA?AV?$vector@H$03@@V1@@Z"(<4 x i32> %p1) #0 {
  %1 = extractelement <4 x i32> %p1, i64 0
  %2 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %1, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %3 = insertelement <4 x i32> undef, i32 %2, i64 0
  %4 = extractelement <4 x i32> %p1, i64 1
  %5 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %4, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %6 = insertelement <4 x i32> %3, i32 %5, i64 1
  %7 = extractelement <4 x i32> %p1, i64 2
  %8 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %7, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %9 = insertelement <4 x i32> %6, i32 %8, i64 2
  %10 = extractelement <4 x i32> %p1, i64 3
  %11 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %10, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %12 = insertelement <4 x i32> %9, i32 %11, i64 3
  %13 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  ret <4 x i32> %12
}

; Function Attrs: nounwind
declare i32 @dx.op.waveActiveOp.i32(i32, i32, i8, i8) #0

attributes #0 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.entryPoints = !{!4}

!0 = !{!"dxc(private) 1.8.0.14796 (main, c2ed9ad4e)"}
!1 = !{i32 1, i32 3}
!2 = !{i32 1, i32 9}
!3 = !{!"lib", i32 6, i32 3}
!4 = !{null, !"", null, null, !5}
!5 = !{i32 0, i64 8912896}
"""

shader2 = """;
; Note: shader requires additional functionality:
;       Wave level operations
;
; shader hash: 7de4736da5ab5cf3ec23ba2d80dea4f8
;
; Buffer Definitions:
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

@dx.nothing.a = internal constant [1 x i32] zeroinitializer

; Function Attrs: nounwind
define <4 x i32> @"\01?fn@@YA?AV?$vector@H$03@@V1@@Z"(<4 x i32> %p1) #0 {
  %1 = extractelement <4 x i32> %p1, i64 0
  %2 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %1, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %3 = insertelement <4 x i32> undef, i32 %2, i64 0
  %4 = extractelement <4 x i32> %p1, i64 1
  %5 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %4, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %6 = insertelement <4 x i32> %3, i32 %5, i64 1
  %7 = extractelement <4 x i32> %p1, i64 2
  %8 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %7, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %9 = insertelement <4 x i32> %6, i32 %8, i64 2
  %10 = extractelement <4 x i32> %p1, i64 3
  %11 = call i32 @dx.op.waveActiveOp.i32(i32 119, i32 %10, i8 3, i8 0)  ; WaveActiveOp(value,op,sop)
  %12 = insertelement <4 x i32> %9, i32 %11, i64 3
  %13 = load i32, i32* getelementptr inbounds ([1 x i32], [1 x i32]* @dx.nothing.a, i32 0, i32 0)
  ret <4 x i32> %12
}

; Function Attrs: nounwind
declare i32 @dx.op.waveActiveOp.i32(i32, i32, i8, i8) #0

attributes #0 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.entryPoints = !{!4}

!0 = !{!"dxc(private) 1.8.0.14796 (main, c2ed9ad4e)"}
!1 = !{i32 1, i32 8}
!2 = !{i32 1, i32 9}
!3 = !{!"lib", i32 6, i32 8}
!4 = !{null, !"", null, null, !5}
!5 = !{i32 0, i64 8912896}
"""

# Diff the function bodies
#diff_result = diff_function_bodies(shader1, shader2)
#if len(diff_result) == 0:
#    print('No Diff Found')
#print(diff_result)
