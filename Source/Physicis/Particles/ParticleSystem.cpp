#include "ParticleSystem.h"
#include <Scene/Scene.h>

ParticleSystem::ParticleSystem(Scene* scene)
	: Component(),
	  bInitialized(false),
	  iCurReadBuffer(0),
	  m_scene(scene)
{
}

ParticleSystem::~ParticleSystem()
{}

bool ParticleSystem::InitalizeParticleSystem()
{
	if(bInitialized)return false;

	Q_ASSERT(initializeOpenGLFunctions());	

	const char* sVaryings[NUM_PARTICLE_ATTRIBUTES] = 
	{
		"vPositionOut",
		"vVelocityOut",
		"vColorOut",
		"fLifeTimeOut",
		"fSizeOut",
		"iTypeOut",
	};

	// Updating program
	spUpdateParticles = new QOpenGLShaderProgram();
	spUpdateParticles->addShaderFromSourceFile(QOpenGLShader::Vertex, "../Resource/Shaders/particles_update.vert");
	spUpdateParticles->addShaderFromSourceFile(QOpenGLShader::Geometry, "../Resource/Shaders/particles_update.geom");
	for (int i = 0; i < NUM_PARTICLE_ATTRIBUTES; ++i)
	{
		glTransformFeedbackVaryings(spUpdateParticles->programId(), 6, sVaryings, GL_INTERLEAVED_ATTRIBS);
	}
	spUpdateParticles->link();

	// Rendering program
	spRenderParticles = new QOpenGLShaderProgram();
	spRenderParticles->addShaderFromSourceFile(QOpenGLShader::Vertex, "../Resource/Shaders/particles_render.vert");
	spRenderParticles->addShaderFromSourceFile(QOpenGLShader::Geometry, "../Resource/Shaders/particles_render.geom");
	spRenderParticles->addShaderFromSourceFile(QOpenGLShader::Fragment, "../Resource/Shaders/particles_render.frag");
	spRenderParticles->link();

	glGenTransformFeedbacks(1, &uiTransformFeedbackBuffer);
	glGenQueries(1, &uiQuery);

	glGenBuffers(2, uiParticleBuffer);
	glGenVertexArrays(2, uiVAO);

	CParticle partInitialization;
	partInitialization.iType = PARTICLE_TYPE_GENERATOR;

	for (int i = 0; i < 2; ++i)
	{
		glBindVertexArray(uiVAO[i]);
		glBindBuffer(GL_ARRAY_BUFFER, uiParticleBuffer[i]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(CParticle)*MAX_PARTICLES_ON_SCENE, NULL, GL_DYNAMIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(CParticle), &partInitialization);

		for (int j = 0; j < NUM_PARTICLE_ATTRIBUTES; ++j)
		{
			glEnableVertexAttribArray(j);
		}


		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)0); // Position
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)12); // Velocity
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)24); // Color
		glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)36); // Lifetime
		glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(CParticle), (const GLvoid*)40); // Size
		glVertexAttribPointer(5, 1, GL_INT,	  GL_FALSE, sizeof(CParticle), (const GLvoid*)44); // Type
	}

	iCurReadBuffer = 0;
	iNumParticles = 1;

	bInitialized = true;

	// texture
	m_Texture = m_scene->textureManager()->addTexture("particle", "../Resource/Textures/particle.bmp");

	return true;
}

float grandf(float fMin, float fAdd)
{
	float fRandom = float(rand()%(RAND_MAX+1))/float(RAND_MAX);
	return fMin+fAdd*fRandom;
}

void ParticleSystem::UpdateParticles( float fTimePassed )
{
	if(!bInitialized)return;
	spUpdateParticles->bind();

	vec3 vUpload;
	spUpdateParticles->setUniformValue("fTimePassed", fTimePassed);
	spUpdateParticles->setUniformValue("vGenPosition", vGenPosition);
	spUpdateParticles->setUniformValue("vGenVelocityMin", vGenVelocityMin);
	spUpdateParticles->setUniformValue("vGenVelocityRange", vGenVelocityRange);
	spUpdateParticles->setUniformValue("vGenColor", vGenColor);
	spUpdateParticles->setUniformValue("vGenGravityVector", vGenGravityVector);

	spUpdateParticles->setUniformValue("fGenLifeMin", fGenLifeMin);
	spUpdateParticles->setUniformValue("fGenLifeRange", fGenLifeRange);

	spUpdateParticles->setUniformValue("fGenLifeMin", fGenLifeMin);
	spUpdateParticles->setUniformValue("fGenLifeRange", fGenLifeRange);

	spUpdateParticles->setUniformValue("fGenSize", fGenSize);
	spUpdateParticles->setUniformValue("iNumToGenerate", 0);

	fElapsedTime += fTimePassed;

	if (fElapsedTime > fNextGenerationTime)
	{
		spUpdateParticles->setUniformValue("iNumToGenerate", iNumToGenerate);
		fElapsedTime -= fNextGenerationTime;

		vec3 vRandomSeed = vec3(grandf(-10.0f, 20.0f), grandf(-10.0f, 20.0f), grandf(-10.0f, 20.0f));
		spUpdateParticles->setUniformValue("vRandomSeed", vRandomSeed);
	}

	glEnable(GL_RASTERIZER_DISCARD);
	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, uiTransformFeedbackBuffer);

	glBindVertexArray(uiVAO[iCurReadBuffer]);
	glEnableVertexAttribArray(1); // Re-enable velocity

	glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, uiParticleBuffer[1-iCurReadBuffer]);

	glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, uiQuery);
	glBeginTransformFeedback(GL_POINTS);

	glDrawArrays(GL_POINTS, 0, iNumParticles);

	glEndTransformFeedback();

	glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
	glGetQueryObjectiv(uiQuery, GL_QUERY_RESULT, &iNumParticles);

	iCurReadBuffer = 1-iCurReadBuffer;

	glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
}

void ParticleSystem::RenderParticles()
{
	if(!bInitialized)return;

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glDepthMask(0);

	glDisable(GL_RASTERIZER_DISCARD);
	spRenderParticles->bind();
	m_Texture->bind(COLOR_TEXTURE_UNIT);
	mat4 matProjection = m_scene->getCamera()->projectionMatrix();
	mat4 matView = m_scene->getCamera()->viewMatrix();

	spRenderParticles->setUniformValue("matrices.mProj", matProjection);
	spRenderParticles->setUniformValue("matrices.mView", matView);
	spRenderParticles->setUniformValue("vQuad1", vQuad1);
	spRenderParticles->setUniformValue("vQuad2", vQuad2);
	spRenderParticles->setUniformValue("gSampler", 0);

	glBindVertexArray(uiVAO[iCurReadBuffer]);
	glDisableVertexAttribArray(1); // Disable velocity, because we don't need it for rendering

	glDrawArrays(GL_POINTS, 0, iNumParticles);

	glDepthMask(1);	
	glDisable(GL_BLEND);
}

void ParticleSystem::SetMatrices()
{
	vQuad1 = vec3::crossProduct(m_scene->getCamera()->viewVector(), m_scene->getCamera()->upVector()).normalized();
	vQuad2 = vec3::crossProduct(m_scene->getCamera()->viewVector(), vQuad1).normalized();
}

void ParticleSystem::SetGeneratorProperties( vec3 a_vGenPosition, vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float fEvery, int a_iNumToGenerate )
{
	vGenPosition = a_vGenPosition;
	vGenVelocityMin = a_vGenVelocityMin;
	vGenVelocityRange = a_vGenVelocityMax - a_vGenVelocityMin;

	vGenGravityVector = a_vGenGravityVector;
	vGenColor = a_vGenColor;
	fGenSize = a_fGenSize;

	fGenLifeMin = a_fGenLifeMin;
	fGenLifeRange = a_fGenLifeMax - a_fGenLifeMin;

	fNextGenerationTime = fEvery;
	fElapsedTime = 0.8f;

	iNumToGenerate = a_iNumToGenerate;
}

int ParticleSystem::GetNumParticles()
{
	return iNumParticles;
}

