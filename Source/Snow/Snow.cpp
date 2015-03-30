#include "Snow.h"
#include "Cuda/Functions.h"
#include "SnowSimulator.h"
#include <Scene/Scene.h>
#include <Utility/Math.h>

Snow::Snow()
	: Component(1),// get rendered last
	  m_particleVBO(0),
	  m_gridVBO(0),
	  m_VAO(0),
	  m_gridSize(0),
	  m_cellSize(0.02f),
	  m_particleCount(200000),
	  m_density(300.0f),
	  m_snowMaterial(1)
{
	m_particles.resize(m_particleCount);
	Q_ASSERT(initializeOpenGLFunctions());
}

Snow::Snow( float cellSize, uint particleCount, float density, uint snowMaterial )
	:Component(1),// get rendered last
	m_particleVBO(0),
	m_gridVBO(0),
	m_VAO(0),
	m_gridSize(0),
	m_cellSize(cellSize),
	m_particleCount(particleCount),
	m_density(density),
	m_snowMaterial(snowMaterial)
{
	m_particles.resize(m_particleCount);
	Q_ASSERT(initializeOpenGLFunctions());
}

Snow::Snow( uint particleCount )
	: Component(1),// get rendered last
	m_particleVBO(0),
	m_gridVBO(0),
	m_VAO(0),
	m_gridSize(0),
	m_cellSize(0.02f),
	m_particleCount(particleCount),
	m_density(100.0f),
	m_snowMaterial(1)
{
	m_particles.resize(m_particleCount);
	Q_ASSERT(initializeOpenGLFunctions());
}

Snow::~Snow()
{
	deleteParticleBuffers();
	deleteGridBuffers();
}

void Snow::initialize()
{
	if (!m_actor)
	{
		qWarning() << "Snow component initialize failed. Needs to be attached to a Game Object first.";
		return;
	}
	installShader();
	voxelizeMesh();
	buildParticleBuffers();
}

void Snow::resetParticleBuffers()
{
	deleteParticleBuffers();
	buildParticleBuffers();
}

void Snow::installShader()
{
	m_renderingEffect = ParticleTechniquePtr(new ParticleTechnique("snow", ParticleTechnique::GENERAL));
	if (!m_renderingEffect->init()) 
	{
		qWarning() << "Snow rendering effect initializing failed.";
		return;
	}
}

void Snow::render( const float currentTime )
{
	m_renderingEffect->enable();
	// the absolute position is already set in the buffer, so there is no need to multiply by the transform matrix
	m_renderingEffect->getShaderProgram()->setUniformValue("gWVP", 
		Scene::instance()->getCamera()->viewProjectionMatrix() /** m_actor->getTransformMatrix()*/);

 	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable( GL_POINT_SMOOTH );

	glBindVertexArray( m_VAO );
	glDrawArrays( GL_POINTS, 0, m_particles.size() );
	glBindVertexArray( 0 );

	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glDisable( GL_POINT_SMOOTH );
}

void Snow::clear()
{
	m_particles.clear();
	deleteParticleBuffers();
}

bool Snow::hasParticleBuffers() const
{
	return m_particleVBO > 0;
}

void Snow::deleteParticleBuffers()
{
	// Delete OpenGL VBO and unregister with CUDA
	if ( hasParticleBuffers() ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_particleVBO );
		glDeleteBuffers( 1, &m_particleVBO );
		glDeleteVertexArrays( 1, &m_VAO );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}
	m_particleVBO = 0;
	m_VAO = 0;
}

void Snow::buildParticleBuffers()
{
	// Build OpenGL VAO
	glGenVertexArrays( 1, &m_VAO );
	glBindVertexArray( m_VAO );

	// Build OpenGL VBO
	glGenBuffers( 1, &m_particleVBO );
	glBindBuffer( GL_ARRAY_BUFFER, m_particleVBO );
	glBufferData( GL_ARRAY_BUFFER, m_particles.size()*sizeof(SnowParticle), m_particles.data(), GL_DYNAMIC_DRAW );

	std::size_t offset = 0; // offset within particle struct

	// Position attribute
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(CUDAVec3);

	// Velocity attribute
	glEnableVertexAttribArray( 1 );
	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(CUDAVec3);

	// Mass attribute
	glEnableVertexAttribArray( 2 );
	glVertexAttribPointer( 2, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(GLfloat);

	// Volume attribute
	glEnableVertexAttribArray( 3 );
	glVertexAttribPointer( 3, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset );
	offset += sizeof(GLfloat);
	offset += 2*sizeof(CUDAMat3);

	// lambda (stiffness) attribute
	offset += 2*sizeof(GLfloat); // skip to material.xi
	glEnableVertexAttribArray(4);
	glVertexAttribPointer( 4, 1, GL_FLOAT, GL_FALSE, sizeof(SnowParticle), (void*)offset);
	offset += sizeof(GLfloat);

	glBindVertexArray( 0 );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	if (m_renderingEffect)
		m_renderingEffect->setVAO(m_VAO);
}

void Snow::voxelizeMesh()
{
	// get the model from the game object
	ModelPtr model = m_actor->getComponent("Model").dynamicCast<IModel>();
	if (!model)
	{
		qWarning() << "Snow component initialize failed. Needs a mesh representation.";
		return;
	}

// 	cudaGraphicsResource* cudaVBO = model->getCudaVBO();
// 	if (!cudaVBO)
// 	{
// 		qWarning() << "Snow component filling mesh failed. CUDA Graphics Resource hasn't been created for the mesh.";
// 		return;
// 	}

	setGrid(model->getBoundingBox()->getGeometryShape().toGrid(m_cellSize));

	//fillMeshWithVBO(&cudaVBO, model->getNumFaces(), m_grid, m_particles.data(), m_particleCount, m_density, m_snowMaterial);
	fillMeshWithTriangles(model->getCudaTriangles().data(), model->getCudaTriangles().size(), m_grid, 
		m_particles.data(), m_particleCount, m_density, m_snowMaterial);

	// if the voxelization is ok, hide the model
	model->setRenderLayer(-1);

	// sync the position of the game object
// 	mat4 transformMatrix = m_actor->getTransformMatrix();
// 	glm::mat4 ctm = Math::Converter::toGLMMat4(transformMatrix);
// 	glm::vec4 newGridpos = ctm * glm::vec4( glm::vec3(m_grid.pos), 0.f );
// 	m_grid.pos = CUDAVec3(newGridpos.x, newGridpos.y, newGridpos.z);

	CUDAVec3 gameobjectPos(m_actor->position().x(), m_actor->position().y(), m_actor->position().z());
	m_grid.pos += gameobjectPos;
	for (int i = 0; i < m_particles.size(); ++i)
	{
		m_particles[i].position += gameobjectPos;
	}

	// add this instance to the simulator
	//SnowSimulator::instance()->addSnowInstance(*this);
	SnowSimulator::instance()->setSnowInstance(this);
	//SnowSimulator::instance()->setGrid(m_grid);

// 	BoxShape box(vec3(0, 3, 0), vec3(3, 3, 3));
// 	setGrid(box.toGrid(0.1f));
	SnowSimulator::instance()->setGrid(m_grid);
}

bool Snow::hasGridBuffers() const
{
	return m_gridVBO > 0;
}

void Snow::deleteGridBuffers()
{
	// Delete OpenGL VBO and unregister with CUDA
	if ( hasGridBuffers() ) 
	{
		glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
		glDeleteBuffers( 1, &m_gridVBO );
		glBindBuffer( GL_ARRAY_BUFFER, 0 );
	}

	m_gridVBO = 0;
}

void Snow::buildGridBuffers()
{
	Node *data = new Node[m_gridSize];
	memset( data, 0, m_gridSize*sizeof(Node) );

	// Build VBO
	glGenBuffers( 1, &m_gridVBO );
	glBindBuffer( GL_ARRAY_BUFFER, m_gridVBO );
	glBufferData( GL_ARRAY_BUFFER, m_gridSize*sizeof(Node), data, GL_DYNAMIC_DRAW );

	delete [] data;

// 	// Mass attribute
// 	glEnableVertexAttribArray( 0 );
// 	glVertexAttribPointer( 0, 1, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(0) );
// 
// 	// Velocity attribute
// 	glEnableVertexAttribArray( 1 );
// 	glVertexAttribPointer( 1, 3, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(sizeof(GLfloat)) );
// 
// 	// Force attribute
// 	glEnableVertexAttribArray( 2 );
// 	glVertexAttribPointer( 2, 3, GL_FLOAT, GL_FALSE, sizeof(Node), (void*)(sizeof(GLfloat)+2*sizeof(vec3)) );

	glBindBuffer( GL_ARRAY_BUFFER, 0 );
}

void Snow::setGrid( const Grid &grid )
{
	m_grid = grid;
	m_gridSize = m_grid.nodeCount();
	resetGridBuffers();
}

void Snow::resetGridBuffers()
{
	deleteGridBuffers();
	buildGridBuffers();
}

Snow& Snow::operator+=( const Snow &other )
{
	m_particles += other.m_particles; 
	resetParticleBuffers(); 
	m_grid = other.m_grid;
	m_gridSize = other.m_gridSize;
	m_gridVBO = other.m_gridVBO;
	return *this;
}

Snow& Snow::operator+=( const SnowParticle &particle )
{
	m_particles.append(particle); 
	resetParticleBuffers(); 
	return *this;
}
