#include "Skybox.h"
#include <Scene/Scene.h>

Skybox::Skybox( Scene* scene )
	: m_scene(scene),
	  m_skyboxTechnique(0),
	  m_cubemapTex(0)
{
}


Skybox::~Skybox()
{
	SAFE_DELETE(m_skyboxTechnique);
	SAFE_DELETE(m_cubemapTex);
}


bool Skybox::init(const QString& PosXFilename,
				  const QString& NegXFilename,
				  const QString& PosYFilename,
				  const QString& NegYFilename,
				  const QString& PosZFilename,
				  const QString& NegZFilename)
{
	Q_ASSERT(initializeOpenGLFunctions());
	m_skyboxTechnique = new SkyboxTechnique();
	if (!m_skyboxTechnique->init()) 
	{
		qWarning() << "Error initializing the skybox technique";
		return false;
	}

	m_cubemapTex = new CubemapTexture(PosXFilename,
									  NegXFilename,
									  PosYFilename,
									  NegYFilename,
									  PosZFilename,
									  NegZFilename);

	ModelLoader loader;
	QVector<ModelDataPtr> modelDataVector = loader.loadModel("../Resource/Models/Common/sphere.obj", false);
	m_vao = loader.getVAO();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];
		// deal with the mesh
		MeshPtr mesh = m_scene->meshManager()->getMesh(data->meshData.name);
		if (!mesh)
		{
			mesh = m_scene->meshManager()->addMesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, 	data->meshData.baseIndex);
		}

		m_meshes.push_back(mesh);
	}

	return true;
}

void Skybox::render( const float currentTime )
{
	m_skyboxTechnique->enable();
	QMatrix4x4 modelMatrix;
	modelMatrix.scale(20, 20, 20);
	QMatrix4x4 modelViewMatrix = m_scene->getCamera()->viewMatrix() * modelMatrix;

	GLint OldCullFaceMode;
	glGetIntegerv(GL_CULL_FACE_MODE, &OldCullFaceMode);
	GLint OldDepthFuncMode;
	glGetIntegerv(GL_DEPTH_FUNC, &OldDepthFuncMode);

	glCullFace(GL_FRONT);
	glDepthFunc(GL_LEQUAL);

	m_skyboxTechnique->setWVP(m_scene->getCamera()->projectionMatrix() * modelViewMatrix);
	m_cubemapTex->bind(GL_TEXTURE0);

	for(int i = 0; i < m_meshes.size(); ++i)
	{
		drawElements(i);
	}

	glCullFace(OldCullFaceMode);        
	glDepthFunc(OldDepthFuncMode);
}

void Skybox::drawElements(uint index)
{
	glBindVertexArray(m_vao);

	glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);

	// Make sure the VAO is not changed from the outside    
	glBindVertexArray(0);
}
