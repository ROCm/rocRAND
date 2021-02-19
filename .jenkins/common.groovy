// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false, boolean addressSanitizer=false)
{
    project.paths.construct_build_prefix()
        
    project.paths.build_command = './install -c'
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String buildStatic = staticLibrary ? '-DBUILD_STATIC_LIBS=ON' : '-DBUILD_SHARED=OFF'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    String addressSanitizerFlags = addressSanitizer ? "CFLAGS='-fsanitize=address -shared-libasan' CXXFLAGS='-fsanitize=address -shared-libasan' LDFLAGS='-fuse-ld=lld'": ""
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${addressSanitizerFlags} ${cmake} -DCMAKE_C_COMPILER=/opt/ropcm/bin/hipcc -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ${buildTypeArg} ${buildStatic} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """
    
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, boolean debug=false, boolean addressSanitizer=false)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''
    String buildTypeDir = debug ? 'debug' : 'release'
    String sanitizerLibPath = addressSanitizer ? project.paths.sanitizerLibPath : ''
    def testCommand = "ctest${centos} --output-on-failure"

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${buildTypeDir}
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/:${sanitizerLibPath} ${testCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project, boolean debug=false)
{
    String buildTypeDir = debug ? 'debug' : 'release'
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/${buildTypeDir}") 
        
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this

