// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, settings)
{
    project.paths.construct_build_prefix()

    project.paths.build_command = './install -c'
    String buildTypeArg = settings.debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = settings.debug ?: 'release'
    String buildStatic = settings.staticLibrary ? '-DBUILD_STATIC_LIBS=ON' : '-DBUILD_SHARED=OFF'
    String codeCovFlag = settings.codeCoverage ? '-DCODE_COVERAGE=ON' : ''
    String asanFlag = settings.addressSanitizer ? '-DBUILD_ADDRESS_SANITIZER=ON' : ''
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''

    String xnackToggle = settings.addressSanitizer ? "export HSA_XNACK=1" : ''

    def command = """#!/usr/bin/env bash
                set -x
                ${xnackToggle}
                rocminfo
                cd ${project.paths.project_build_prefix}
                # gfxTargetParser reads gfxarch and adds target features such as xnack
                ${auxiliary.gfxTargetParser()}
                sudo ./install -ci --address-sanitizer
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, settings)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    // String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''
    // Disable xorwow test for now as it is a known failure with gfx90a.
    // def testCommand = "ctest${centos} --output-on-failure"
    def testCommand = "ctest --output-on-failure"

    String LD_PATH = settings.addressSanitizer ? """export ASAN_LIB_PATH=\$(/opt/rocm/llvm/bin/clang -print-file-name=libclang_rt.asan-x86_64.so)
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(dirname "\${ASAN_LIB_PATH}")""" : 'export LD_LIBRARY_PATH=/opt/rocm/lib/'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make -j4
                ${LD_PATH}
                ldd ctest
                ${sudo} ${testCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

def runCodeCovTestCommand(platform, project, jobName)
{
    withCredentials([string(credentialsId: 'mathlibs-codecov-token-rocrand', variable: 'CODECOV_TOKEN')])
    {
        String prflag = env.CHANGE_ID ? "--pr \"${env.CHANGE_ID}\"" : ''

        String objectFlags = "-object ./library/librocrand.so"

        String profdataFile = "./rocRand.profdata"
        String reportFile = "./code_cov_rocRand.report"
        String coverageFile = "./code_cov_rocRand.txt"
        String coverageFilter = "(.*googletest-src.*)|(.*/yaml-cpp-src/.*)|(.*hip/include.*)|(.*/include/llvm/.*)|(.*test/unit.*)|(.*/spdlog/.*)|(.*/msgpack-src/.*)"

        def command = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}/build/release
                    #Remove any preexisting prof files.
                    rm -rf ./test/*.profraw

                    #The `%m` creates a different prof file for each object file.
                    LLVM_PROFILE_FILE=./rocRand_%m.profraw ctest --output-on-failure

                    #this combines them back together.
                    /opt/rocm/llvm/bin/llvm-profdata merge -sparse ./test/*.profraw -o ${profdataFile}

                    #For some reason, with the -object flag, we can't just specify the source directory, so we have to filter out the files we don't want.
                    /opt/rocm/llvm/bin/llvm-cov report ${objectFlags} -instr-profile=${profdataFile} -ignore-filename-regex="${coverageFilter}" > ${reportFile}
                    cat ${reportFile}
                    /opt/rocm/llvm/bin/llvm-cov show -Xdemangler=/opt/rocm/llvm/bin/llvm-cxxfilt ${objectFlags} -instr-profile=${profdataFile} -ignore-filename-regex="${coverageFilter}" > ${coverageFile}

                    #Upload report to codecov
                    curl -Os https://uploader.codecov.io/latest/linux/codecov
                    chmod +x codecov
                    ./codecov -t ${CODECOV_TOKEN} ${prflag} --flags "${platform.gpu}" --sha \$(git rev-parse HEAD) --name "CI: ${jobName}" --file ${coverageFile} -v
                """
        platform.runCommand(this, command)
    }
}

return this
