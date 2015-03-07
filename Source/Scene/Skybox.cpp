#include "Skybox.h"
#include <Scene/Scene.h>

Skybox::Skybox()
	: m_scene(Scene::instance()),
	  m_skyboxTechnique(SkyboxTechniquePtr()),
	  m_cubemapTex(0)
{
}


Skybox::~Skybox()
{
	// clean up the meshes
	foreach(MeshPtr mesh, m_meshes)
	{
		MeshManager::instance()->deleteMesh(mesh);
		mesh.clear();
	}
}


bool Skybox::init(const QString& PosXFilename,
				  const QString& NegXFilename,
				  const QString& PosYFilename,
				  const QString& NegYFilename,
				  const QString& PosZFilename,
				  const QString& NegZFilename)
{
	Q_ASSERT(initializeOpenGLFunctions());

	m_skyboxTechnique = SkyboxTechniquePtr(new SkyboxTechnique());
	if (!m_skyboxTechnique->init()) 
	{
		qWarning() << "Error initializing the skybox technique";
		return false;
	}

	m_cubemapTex = CubemapTexturePtr( new CubemapTexture(PosXFilename,
														 NegXFilename,
														 PosYFilename,
														 NegYFilename,
														 PosZFilename,
														 NegZFilename));

	ModelLoader loader;
	QVector<ModelDataPtr> modelDataVector = loader.loadModel("../Resource/Models/Common/sphere.obj", m_skyboxTechnique->getShaderProgram()->programId());
	m_vao = loader.getVAO();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];
		// deal with the mesh
		MeshPtr mesh = MeshManager::instance()->getMesh(data->meshData.name);
		if (!mesh)
		{
			mesh = MeshManager::instance()->addMesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, 	data->meshData.baseIndex);
		}

		m_meshes.push_back(mesh);
	}

	return true;
}

void Skybox::render( const float currentTime )
{
	m_skyboxTechnique->enable();

	mat4 modelMatrix;
	modelMatrix.translate(m_scene->getCamera()->position());

	GLint OldCullFaceMode;
	glGetIntegerv(GL_CULL_FACE_MODE, &OldCullFaceMode);
	GLint OldDepthFuncMode;
	glGetIntegerv(GL_DEPTH_FUNC, &OldDepthFuncMode);

	glCullFace(GL_FRONT);
	glDepthFunc(GL_LEQUAL);

	m_skyboxTechnique->setMVPMatrix(m_scene->getCamera()->viewProjectionMatrix() * modelMatrix);
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
