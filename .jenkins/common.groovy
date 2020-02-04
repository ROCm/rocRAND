// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()
        
    rocrand.paths.build_command = jobName.contains('hipclang') ? './install -c --hip-clang' : './install -c'
    rocrand.compiler.compiler_path = platform.jenkinsLabel.contains('hip-clang') ? '/opt/rocm/bin/hipcc' : '/opt/rocm/bin/hcc'        

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib/ CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """
    
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make -j4
                ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/ ctest --output-on-failure
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

