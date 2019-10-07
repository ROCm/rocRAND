#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocRANDCI:
{

    def rocrand = new rocProject('rocRAND')

    def nodes = new dockerNodes(['ubuntu && gfx803', 'gfx900 && centos7', 'gfx906 && centos7', 'sles && gfx906', 'gfx908 && ubuntu'], rocrand)

    boolean formatCheck = false
     
    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        rocrand.paths.build_command = platform.jenkinsLabel.contains('hip-clang') ? './install -c --hip-clang' : './install -c'
        rocrand.compiler.compiler_path = platform.jenkinsLabel.contains('hip-clang') ? '/opt/rocm/bin/hipcc' : '/opt/rocm/bin/hcc'        

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib/ CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                    """
        
        platform.runCommand(this, command)
    }

    
    def testCommand =
    {
        platform, project->
        
        String sudo = auxiliary.sudo(platform.jenkinsLabel)

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/ ctest --output-on-failure
                """

        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->
        
        def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release") 
        
        platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
    }

    buildProject(rocrand, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

