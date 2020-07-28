// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false)
{
    project.paths.construct_build_prefix()
        
    project.paths.build_command = jobName.contains('hipclang') ? './install -c --hip-clang' : './install -c'
    project.compiler.compiler_path = platform.jenkinsLabel.contains('hip-clang') ? '/opt/rocm/bin/hipcc' : '/opt/rocm/bin/hcc'        
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String buildStatic = staticLibrary ? '-DBUILD_SHARED=OFF' : '-DBUILD_SHARED=ON'
    
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib/ CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/${compiler} ${buildTypeArg} ${buildStatic} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """
    
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''
    def testCommand = "ctest${centos} --output-on-failure"

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make -j4
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/ ${testCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release") 
        
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this

