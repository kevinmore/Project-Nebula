#include "ParticleSystem.h"
#include <Scene/Scene.h>
#include <Utility/Math.h>

ParticleSystem::ParticleSystem(Scene* scene)
	: Component(true, 1),// get rendered last
	  bInitialized(false),
	  iCurReadBuffer(0),
	  fElapsedTime(0.0f),
	  m_scene(scene)
{
}

ParticleSystem::~ParticleSystem()
{}

bool ParticleSystem::initParticleSystem()
{
	if(bInitialized)
		return false;

	Q_ASSERT(initializeOpenGLFunctions());	

	installShaders();

	prepareTransformFeedback();


	iCurReadBuffer = 0;
	iNumParticles = 1;

	bInitialized = true;

	// load texture
	m_Texture = m_scene->textureManager()->addTexture("particle", "../Resource/Textures/particle.bmp");

	return true;
}


void ParticleSystem::installShaders()
{
	// Updating program
	spUpdateParticles = new ParticleTechnique("particles_update", ParticleTechnique::UPDATE);
	if (!spUpdateParticles->init()) 
	{
		qWarning() << "particles_update initializing failed.";
		return;
	}

	// Rendering program
	spRenderParticles = new ParticleTechnique("particles_render", ParticleTechnique::RENDER);
	if (!spRenderParticles->init()) 
	{
		qWarning() << "particles_render initializing failed.";
		return;
	}
}

void ParticleSystem::prepareTransformFeedback()
{
	glGenTransformFeedbacks(1, &uiTransformFeedbackBuffer);
	glGenQueries(1, &uiQuery);

	glGenVertexArrays(2, uiVAO);
	glGenBuffers(2, uiParticleBuffer);

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

	spUpdateParticles->setVAO(uiVAO[0]);
	spRenderParticles->setVAO(uiVAO[1]);
}


void ParticleSystem::updateParticles( float fTimePassed )
{
	if(!bInitialized)return;
	spUpdateParticles->enable();
	vGenPosition = m_actor->position();

	spUpdateParticles->getShaderProgram()->setUniformValue("fTimePassed", fTimePassed);
	spUpdateParticles->getShaderProgram()->setUniformValue("vGenPosition", vGenPosition);
	spUpdateParticles->getShaderProgram()->setUniformValue("vGenVelocityMin", vGenVelocityMin);
	spUpdateParticles->getShaderProgram()->setUniformValue("vGenVelocityRange", vGenVelocityRange);
	spUpdateParticles->getShaderProgram()->setUniformValue("vGenColor", vGenColor);
	spUpdateParticles->getShaderProgram()->setUniformValue("vGenGravityVector", vGenGravityVector);

	spUpdateParticles->getShaderProgram()->setUniformValue("fGenLifeMin", fGenLifeMin);
	spUpdateParticles->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	spUpdateParticles->getShaderProgram()->setUniformValue("fGenLifeMin", fGenLifeMin);
	spUpdateParticles->getShaderProgram()->setUniformValue("fGenLifeRange", fGenLifeRange);

	spUpdateParticles->getShaderProgram()->setUniformValue("fGenSize", fGenSize);
	spUpdateParticles->getShaderProgram()->setUniformValue("iNumToGenerate", 0);

	if (fElapsedTime > fNextGenerationTime)
	{
		spUpdateParticles->getShaderProgram()->setUniformValue("iNumToGenerate", iNumToGenerate);
		fElapsedTime -= fNextGenerationTime;

		vec3 vRandomSeed =  Math::Random::randUnitVec3();
		spUpdateParticles->getShaderProgram()->setUniformValue("vRandomSeed", vRandomSeed);
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

void ParticleSystem::render(const float currentTime)
{
	if(!bInitialized)return;

	float dt = currentTime - fElapsedTime;
	fElapsedTime = currentTime;

	updateParticles(dt);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glDepthMask(0);

	glDisable(GL_RASTERIZER_DISCARD);
	spRenderParticles->enable();
	m_Texture->bind(COLOR_TEXTURE_UNIT);

	mat4 matProjection = m_scene->getCamera()->projectionMatrix();
	mat4 matView = m_scene->getCamera()->viewMatrix();
	vQuad1 = vec3::crossProduct(m_scene->getCamera()->viewVector(), m_scene->getCamera()->upVector()).normalized();
	vQuad2 = vec3::crossProduct(m_scene->getCamera()->viewVector(), vQuad1).normalized();

	spRenderParticles->getShaderProgram()->setUniformValue("matrices.mProj", matProjection);
	spRenderParticles->getShaderProgram()->setUniformValue("matrices.mView", matView);
	spRenderParticles->getShaderProgram()->setUniformValue("vQuad1", vQuad1);
	spRenderParticles->getShaderProgram()->setUniformValue("vQuad2", vQuad2);
	spRenderParticles->getShaderProgram()->setUniformValue("gSampler", COLOR_TEXTURE_UNIT);

	glBindVertexArray(uiVAO[iCurReadBuffer]);
	glDisableVertexAttribArray(1); // Disable velocity, because we don't need it for rendering

	glDrawArrays(GL_POINTS, 0, iNumParticles);
	glDepthMask(1);	
	glDisable(GL_BLEND);
}


void ParticleSystem::setGeneratorProperties( vec3 a_vGenVelocityMin, vec3 a_vGenVelocityMax, vec3 a_vGenGravityVector, vec3 a_vGenColor, float a_fGenLifeMin, float a_fGenLifeMax, float a_fGenSize, float fEvery, int a_iNumToGenerate )
{
	vGenVelocityMin = a_vGenVelocityMin;
	vGenVelocityRange = a_vGenVelocityMax - a_vGenVelocityMin;

	vGenGravityVector = a_vGenGravityVector;
	vGenColor = a_vGenColor;
	fGenSize = a_fGenSize;

	fGenLifeMin = a_fGenLifeMin;
	fGenLifeRange = a_fGenLifeMax - a_fGenLifeMin;

	fNextGenerationTime = fEvery;

	iNumToGenerate = a_iNumToGenerate;
}

int ParticleSystem::getNumParticles()
{
	return iNumParticles;
}
