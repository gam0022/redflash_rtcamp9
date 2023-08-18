/* 
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//-----------------------------------------------------------------------------
//
// optixDemandLoadTexture: simple demonstration of demand loaded textures
//
//-----------------------------------------------------------------------------

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glew.h>
#if defined( _WIN32 )
#include <GL/freeglut.h>
#include <GL/wglew.h>
#else
#include <GL/glut.h>
#endif
#endif

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "common.h"
#include <sutil/Arcball.h>
#include <sutil/sutil.h>

#include <algorithm>
#include <sstream>
#include <string.h>  // for memcpy
// for uint32_t
#include <stdint.h>

using namespace optix;

const char* const SAMPLE_NAME = "optixDemandLoadTexture";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

Context  context;
uint32_t width   = 512u;
uint32_t height  = 512u;
bool     use_pbo = true;
Aabb     aabb;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;

Group top_group;

//------------------------------------------------------------------------------
//
// Forward decls
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer();
void   destroyContext();
void   registerExitHandler();
void   createContext();

void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize( int* argc, char** argv );
void glutRun();

void glutDisplay();
void glutKeyboardPress( unsigned char k, int x, int y );
void glutResize( int w, int h );


//------------------------------------------------------------------------------
//
//  Helper functions
//
//------------------------------------------------------------------------------

Buffer getOutputBuffer()
{
    return context["output_buffer"]->getBuffer();
}


void destroyContext()
{
    if( context )
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
// register shutdown handler
#ifdef _WIN32
    glutCloseFunc( destroyContext );  // this function is freeglut-only
#else
    atexit( destroyContext );
#endif
}


void createContext()
{
    // Set up context
    context = Context::create();
    context->setRayTypeCount( 2 );
    context->setEntryPointCount( 1 );
    context->setExceptionEnabled( RT_EXCEPTION_USER, true );

    context["scene_epsilon"]->setFloat( 1.e-4f );

    Buffer buffer = sutil::createOutputBuffer( context, RT_FORMAT_UNSIGNED_BYTE4, width, height, use_pbo );
    context["output_buffer"]->set( buffer );

    // Ray generation program
    const char* ptx             = sutil::getPtxString( SAMPLE_NAME, "pinhole_camera.cu" );
    Program     ray_gen_program = context->createProgramFromPTXString( ptx, "pinhole_camera" );
    context->setRayGenerationProgram( 0, ray_gen_program );

    // Exception program
    Program exception_program = context->createProgramFromPTXString( ptx, "exception" );
    context->setExceptionProgram( 0, exception_program );
    context["bad_color"]->setFloat( 1.0f, 0.0f, 1.0f );

    // Miss program
    context->setMissProgram(
        0, context->createProgramFromPTXString( sutil::getPtxString( SAMPLE_NAME, "constantbg.cu" ), "miss" ) );
    context["bg_color"]->setFloat( 0.34f, 0.55f, 0.85f );
}

int demandLoadCallback( void* callbackData, RTbuffer buffer, RTmemoryblock* block )
{
    float4 mipLevelColors[] = {
        {255, 0, 0, 0},    // red
        {255, 127, 0, 0},  // orange
        {255, 255, 0, 0},  // yellow
        {0, 255, 0, 0},    // green
        {0, 0, 255, 0},    // blue
        {127, 0, 0, 0},    // dark red
        {127, 63, 0, 0},   // dark orange
        {127, 127, 0, 0},  // dark yellow
        {0, 127, 0, 0},    // dark green
        {0, 0, 127, 0},    // dark blue
    };
    float4 color = mipLevelColors[block->mipLevel] / make_float4( 255, 255, 255, 255 );
    float4 black = make_float4( 0.f );

    unsigned int gridSize = std::max( 1, 16 >> block->mipLevel );
    for( unsigned int y = 0; y < block->height; ++y )
    {
        float4* row = reinterpret_cast<float4*>( static_cast<char*>( block->baseAddress ) + block->rowPitch * y );
        for( unsigned int x = 0; x < block->width; ++x )
        {
            bool a = x / gridSize % 2 != 0;
            bool b = y / gridSize % 2 != 0;
            row[x] = ( a && b ) || ( !a && !b ) ? color : black;
        }
    }
    return 1;
}

// Return T->Geom
Transform makeInstance( const float3& translation )
{
    const char* ptx              = sutil::getPtxString( SAMPLE_NAME, "sphere_texcoord.cu" );
    Program     sphere_bounds    = context->createProgramFromPTXString( ptx, "bounds" );
    Program     sphere_intersect = context->createProgramFromPTXString( ptx, "intersect" );

    // Sphere
    Geometry sphere = context->createGeometry();
    sphere->setPrimitiveCount( 1u );
    sphere->setBoundingBoxProgram( sphere_bounds );
    sphere->setIntersectionProgram( sphere_intersect );
    sphere["sphere"]->setFloat( 0.0f, 1.2f, 0.0f, 1.0f );
    sphere["matrix_row_0"]->setFloat( 1.0f, 0.0f, 0.0f );
    sphere["matrix_row_1"]->setFloat( 0.0f, 1.0f, 0.0f );
    sphere["matrix_row_2"]->setFloat( 0.0f, 0.0f, 1.0f );
    float3 minimum = make_float3( -1.0f, 0.2f, -1.0f );
    float3 maximum = make_float3( 1.0f, 2.2f, 1.0f );
    aabb.set( minimum, maximum );

    Buffer demanded = context->createBufferFromCallback( RT_BUFFER_INPUT, demandLoadCallback, NULL, RT_FORMAT_FLOAT4, 1024, 1024 );
    demanded->setMipLevelCount( 10 );

    TextureSampler sampler = context->createTextureSampler();
    sampler->setBuffer( demanded );
    context["map_id"]->setInt( sampler->getId() );


    Material sphere_matl = context->createMaterial();
    ptx                  = sutil::getPtxString( SAMPLE_NAME, "demandLoad.cu" );
    Program sphere_ch    = context->createProgramFromPTXString( ptx, "closest_hit_radiance_textured" );
    sphere_matl->setClosestHitProgram( 0, sphere_ch );
    sphere_matl["Ka"]->setFloat( 0.3f, 0.3f, 0.3f );
    sphere_matl["Kd"]->setFloat( 0.6f, 0.7f, 0.8f );
    sphere_matl["Ks"]->setFloat( 0.8f, 0.9f, 0.8f );
    sphere_matl["phong_exp"]->setFloat( 88 );
    sphere_matl["reflectivity_n"]->setFloat( 0.2f, 0.2f, 0.2f );

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry( sphere );
    gi->setMaterialCount( 1 );
    gi->setMaterial( 0, sphere_matl );

    GeometryGroup gg = context->createGeometryGroup();
    gg->setChildCount( 1 );
    gg->setChild( 0, gi );
    gg->setAcceleration( context->createAcceleration( "Trbvh" ) );

    context["top_object"]->set( gg );
    context["top_shadower"]->set( gg );

    Transform transform = context->createTransform();
    Matrix<4, 4> m = Matrix<4, 4>::translate( translation );
    transform->setMatrix( false, m.getData(), NULL );
    transform->setChild( gg );

    return transform;
}


void setupCamera()
{
    const float max_dim = fmaxf( aabb.extent( 0 ), aabb.extent( 1 ) );  // max of x, y components

    camera_eye    = aabb.center() + make_float3( 0.0f, max_dim * 1.5f, max_dim * 1.5f );
    camera_lookat = aabb.center();
    camera_up     = make_float3( 0.0f, 1.0f, 0.0f );

    camera_rotate = Matrix4x4::identity();
}


void setupLights()
{
    const float max_dim = fmaxf( aabb.extent( 0 ), aabb.extent( 1 ) );  // max of x, y components

    BasicLight lights[] = {{make_float3( -0.5f, 0.25f, -1.0f ), make_float3( 0.2f, 0.2f, 0.25f ), 0, 0},
                           {make_float3( -0.5f, 0.0f, 1.0f ), make_float3( 0.1f, 0.1f, 0.10f ), 0, 0},
                           {make_float3( 0.5f, 0.5f, 0.5f ), make_float3( 0.7f, 0.7f, 0.65f ), 1, 0}};
    lights[0].pos *= max_dim * 10.0f;
    lights[1].pos *= max_dim * 10.0f;
    lights[2].pos *= max_dim * 10.0f;

    Buffer light_buffer = context->createBuffer( RT_BUFFER_INPUT );
    light_buffer->setFormat( RT_FORMAT_USER );
    light_buffer->setElementSize( sizeof( BasicLight ) );
    light_buffer->setSize( sizeof( lights ) / sizeof( lights[0] ) );
    memcpy( light_buffer->map(), lights, sizeof( lights ) );
    light_buffer->unmap();

    context["lights"]->set( light_buffer );
}


void updateCamera()
{
    const float vfov         = 35.0f;
    const float aspect_ratio = static_cast<float>( width ) / static_cast<float>( height );

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables( camera_eye, camera_lookat, camera_up, vfov, aspect_ratio, camera_u, camera_v, camera_w, true );

    const Matrix4x4 frame =
        Matrix4x4::fromBasis( normalize( camera_u ), normalize( camera_v ), normalize( -camera_w ), camera_lookat );
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

    camera_eye    = make_float3( trans * make_float4( camera_eye, 1.0f ) );
    camera_lookat = make_float3( trans * make_float4( camera_lookat, 1.0f ) );
    camera_up     = make_float3( trans * make_float4( camera_up, 0.0f ) );

    sutil::calculateCameraVariables( camera_eye, camera_lookat, camera_up, vfov, aspect_ratio, camera_u, camera_v, camera_w, true );

    camera_rotate = Matrix4x4::identity();

    context["eye"]->setFloat( camera_eye );
    context["U"]->setFloat( camera_u );
    context["V"]->setFloat( camera_v );
    context["W"]->setFloat( camera_w );
}


void glutInitialize( int* argc, char** argv )
{
    glutInit( argc, argv );
    glutInitDisplayMode( GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE );
    glutInitWindowSize( width, height );
    glutInitWindowPosition( 100, 100 );
    glutCreateWindow( SAMPLE_NAME );
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, 1, 0, 1, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glViewport( 0, 0, width, height );

    glutShowWindow();
    glutReshapeWindow( width, height );

    // register glut callbacks
    glutDisplayFunc( glutDisplay );
    glutIdleFunc( glutDisplay );
    glutReshapeFunc( glutResize );
    glutKeyboardFunc( glutKeyboardPress );

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();
    context->launch( 0, width, height );

    sutil::displayBufferGL( getOutputBuffer() );

    {
        static unsigned frame_count = 0;
        sutil::displayFps( frame_count++ );
    }

    glutSwapBuffers();
}


void glutKeyboardPress( unsigned char k, int x, int y )
{

    switch( k )
    {
        case( 'q' ):
        case( 27 ):  // ESC
        {
            destroyContext();
            exit( 0 );
        }
        case( 's' ):
        {
            const std::string outputImage = std::string( SAMPLE_NAME ) + ".ppm";
            std::cerr << "Saving current frame to '" << outputImage << "'\n";
            sutil::displayBufferPPM( outputImage.c_str(), getOutputBuffer() );
            break;
        }
    }
}


void glutResize( int w, int h )
{
    if( w == (int)width && h == (int)height )
        return;

    width  = w;
    height = h;
    sutil::ensureMinimumSize( width, height );

    sutil::resizeBuffer( getOutputBuffer(), width, height );

    glViewport( 0, 0, width, height );

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cerr << "\nUsage: " << argv0 << " [options]\n";
    std::cerr << "App Options:\n"
                 "  -h | --help               Print this usage message and exit.\n"
                 "  -f | --file               Save single frame to file and exit.\n"
                 "  -n | --nopbo              Disable GL interop for display buffer.\n"
                 "  -d | --dim=<width>x<height> Set image dimensions. Defaults to 512x512.\n"
                 "App Keystrokes:\n"
                 "  q  Quit\n"
                 "  s  Save image to '"
              << SAMPLE_NAME << ".ppm'\n"
              << std::endl;

    exit( 1 );
}

int main( int argc, char** argv )
{
    std::string outFile;
    int         warmup = 0;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "-f" || arg == "--file" )
        {
            if( i == argc - 1 )
            {
                std::cerr << "Option '" << arg << "' requires additional argument.\n";
                printUsageAndExit( argv[0] );
            }
            outFile = argv[++i];
        }
        else if( arg == "-n" || arg == "--nopbo" )
        {
            use_pbo = false;
        }
        else if( arg.find( "-d" ) == 0 || arg.find( "--dim" ) == 0 )
        {
            const size_t index = arg.find_first_of( '=' );
            if( index == std::string::npos )
            {
                std::cerr << "Option '" << arg
                          << "' is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit( argv[0] );
            }
            const std::string dim = arg.substr( index + 1 );
            try
            {
                sutil::parseDimensions( dim.c_str(), (int&)width, (int&)height );
            }
            catch( const Exception& )
            {
                std::cerr << "Option '" << arg
                          << "' is malformed. Please use the syntax -d | --dim=<width>x<height>.\n";
                printUsageAndExit( argv[0] );
            }
        }
        else if( arg.find( "--warmup" ) == 0 )
        {
            const size_t index = arg.find_first_of( '=' );
            if( index == std::string::npos )
            {
                std::cerr << "Option '" << arg << "' is malformed.  Please use the syntax --warmup=<N>.\n";
                printUsageAndExit( argv[0] );
            }
            std::istringstream value( arg.substr( index + 1 ) );
            value >> warmup;
            if( !value )
            {
                std::cerr << "Option '" << arg << "' is malformed.  Please use the syntax --warmup=<N>.\n";
                printUsageAndExit( argv[0] );
            }
        }
        else
        {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        glutInitialize( &argc, argv );

#ifndef __APPLE__
        glewInit();
#endif

        createContext();
        top_group = context->createGroup();
        top_group->setAcceleration( context->createAcceleration( "Trbvh" ) );
        top_group->setChildCount( 1 );
        top_group->setChild( 0, makeInstance( make_float3( 0.0f, 0.0f, 0.0f ) ) );
        context["top_object"]->set( top_group );

        setupCamera();
        setupLights();

        context->validate();

        if( outFile.empty() )
        {
            glutRun();
        }
        else
        {
            updateCamera();
            for( int i = 0; i < warmup; ++i )
                context->launch( 0, width, height );
            context->launch( 0, width, height );
            sutil::displayBufferPPM( outFile.c_str(), getOutputBuffer() );
            destroyContext();
        }
        return 0;
    }
    SUTIL_CATCH( context->get() )
}
