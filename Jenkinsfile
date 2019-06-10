#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocrandCI:
{

    def rocrand = new rocProject('rocrand')
    rocrand.paths.build_command = './install -i'
    def nodes = new dockerNodes(['gfx900', 'gfx906'], rocrand)

    boolean formatCheck = false
     
    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        def command = """#!/usr/bin/env bash
                  set -x
                  cd ${project.paths.project_build_prefix}
                  export PATH=/opt/rocm/bin:$PATH
                  CXX=hcc ${project.paths.build_command}
                """
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def ctest = 'ctest --output-on-failure'

        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}/build/release
                        ${ctest}
                  """

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
                    cd ${project.paths.project_build_prefix}
                    ./install -p
                    cd ${project.paths.project_build_prefix}/build/release
                    rm -rf package && mkdir -p package
                    mv *.rpm package/
                    rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")        
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}
                    ./install -p
                    cd ${project.paths.project_build_prefix}/build/release
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


