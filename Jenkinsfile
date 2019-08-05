#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocRANDCI:
{

    def rocrand = new rocProject('rocRAND')

    def nodes = new dockerNodes(['gfx900 && ubuntu', 'gfx906 && ubuntu', 'gfx900 && centos7', 'gfx906 && centos7'], rocrand)

    boolean formatCheck = false
     
    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command 

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                    """
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
                    """
        }

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('centos'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    sudo ctest --output-on-failure
                """
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    ctest --output-on-failure
                """
        }

        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->

        def command
        
        if(platform.jenkinsLabel.contains('centos'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    rm -rf package && mkdir -p package
                    mv *.rpm package/
                    rpm -qlp package/*.rpm
                  """
            
            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
        }
        else if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make package
                    rm -rf package && mkdir -p package
                    mv *.deb package/
                    dpkg -c package/*.deb
                  """        
            
            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
        }
    }

    buildProject(rocrand, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

