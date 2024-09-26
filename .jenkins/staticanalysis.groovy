#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, settings)
{
    project.paths.construct_build_prefix()
}

def runCI =
{
    def settings = [:]

    nodeDetails, jobName->

    def prj  = new rocProject('rocRAND', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = true

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, settings)
    }

    buildProject(prj , formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
    
    def jobNameList = ["main":([ubuntu22:['any']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList, 'rocBLAS')

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }
}
