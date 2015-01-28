#include <limits.h>
#include <assert.h>
#include "ShadingTechnique.h"
#include <QtMath>
#include <Scene/Scene.h>

#define SNPRINTF _snprintf_s


using namespace std;

ShadingTechnique::ShadingTechnique(Scene* scene, const QString &shaderName, ShaderType shaderType)
	: Technique(shaderName),
	  m_scene(scene),
	  m_shaderType(shaderType)
{
}


bool ShadingTechnique::init()
{
    if (!Technique::init() || m_shaderFileName.isEmpty()) 
	{
        return false;
    }

	return compileShader();
}

bool ShadingTechnique::compileShader()
{
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + m_shaderFileName + ".vert");
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + m_shaderFileName + ".frag");
	m_shaderProgram->link();

	m_numPointLightsLocation = getUniformLocation("gNumPointLights");
	m_numSpotLightsLocation = getUniformLocation("gNumSpotLights");
		
	for (unsigned int i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(m_pointLightsLocation) ; i++) {
		char Name[128];
		memset(Name, 0, sizeof(Name));
		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Base.Color", i);
		m_pointLightsLocation[i].Color = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Base.AmbientIntensity", i);
		m_pointLightsLocation[i].AmbientIntensity = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Position", i);
		m_pointLightsLocation[i].Position = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Base.DiffuseIntensity", i);
		m_pointLightsLocation[i].DiffuseIntensity = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Atten.Constant", i);
		m_pointLightsLocation[i].Atten.Constant = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Atten.Linear", i);
		m_pointLightsLocation[i].Atten.Linear = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gPointLights[%d].Atten.Exp", i);
		m_pointLightsLocation[i].Atten.Exp = getUniformLocation(Name);

		if (m_pointLightsLocation[i].Color == INVALID_LOCATION ||
			m_pointLightsLocation[i].AmbientIntensity == INVALID_LOCATION ||
			m_pointLightsLocation[i].Position == INVALID_LOCATION ||
			m_pointLightsLocation[i].DiffuseIntensity == INVALID_LOCATION ||
			m_pointLightsLocation[i].Atten.Constant == INVALID_LOCATION ||
			m_pointLightsLocation[i].Atten.Linear == INVALID_LOCATION ||
			m_pointLightsLocation[i].Atten.Exp == INVALID_LOCATION) {
				return false;
		}
	}

	for (unsigned int i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(m_spotLightsLocation) ; i++) {
		char Name[128];
		memset(Name, 0, sizeof(Name));
		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Base.Color", i);
		m_spotLightsLocation[i].Color = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Base.AmbientIntensity", i);
		m_spotLightsLocation[i].AmbientIntensity = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Position", i);
		m_spotLightsLocation[i].Position = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Direction", i);
		m_spotLightsLocation[i].Direction = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Cutoff", i);
		m_spotLightsLocation[i].Cutoff = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Base.DiffuseIntensity", i);
		m_spotLightsLocation[i].DiffuseIntensity = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Constant", i);
		m_spotLightsLocation[i].Atten.Constant = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Linear", i);
		m_spotLightsLocation[i].Atten.Linear = getUniformLocation(Name);

		SNPRINTF(Name, sizeof(Name), "gSpotLights[%d].Base.Atten.Exp", i);
		m_spotLightsLocation[i].Atten.Exp = getUniformLocation(Name);

		if (m_spotLightsLocation[i].Color == INVALID_LOCATION ||
			m_spotLightsLocation[i].AmbientIntensity == INVALID_LOCATION ||
			m_spotLightsLocation[i].Position == INVALID_LOCATION ||
			m_spotLightsLocation[i].Direction == INVALID_LOCATION ||
			m_spotLightsLocation[i].Cutoff == INVALID_LOCATION ||
			m_spotLightsLocation[i].DiffuseIntensity == INVALID_LOCATION ||
			m_spotLightsLocation[i].Atten.Constant == INVALID_LOCATION ||
			m_spotLightsLocation[i].Atten.Linear == INVALID_LOCATION ||
			m_spotLightsLocation[i].Atten.Exp == INVALID_LOCATION) {
				return false;
		}
	}

	if (m_shaderType == RIGGED)
	{
		for (unsigned int i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(m_boneLocation) ; i++) {
			char Name[128];
			memset(Name, 0, sizeof(Name));
			SNPRINTF(Name, sizeof(Name), "gBones[%d]", i);
			m_boneLocation[i] = getUniformLocation(Name);
		}
	}



// 	DirectionalLight directionalLight;
// 	directionalLight.Color = vec3(1.0f, 1.0f, 1.0f);
// 	directionalLight.AmbientIntensity = 0.55f;
// 	directionalLight.DiffuseIntensity = 0.9f;
// 	directionalLight.Direction = vec3(-1.0f, -1.0, -1.0);
// 
// 	enable();
// 	setColorTextureUnit(0);
// 	setNormalMapTextureUnit(2);
// 	setDirectionalLight(directionalLight);
// 	setMatSpecularIntensity(1.0f);
// 	setMatSpecularPower(5);

	PointLight pl;
	pl.Position = vec3(100,200,100);
	pl.Color = vec3(1.0f, 1.0f, 1.0f);
	pl.AmbientIntensity = 0.55f;
	pl.DiffuseIntensity = 0.9f;

	enable();
	setPointLights(1, &pl);
	Material mat("11");
	setMaterial(mat);

	initLights();

	return true;
}

void ShadingTechnique::setDirectionalLight(const DirectionalLight& Light)
{
   	m_shaderProgram->setUniformValue("gDirectionalLight.Base.Color", Light.Color);
 	m_shaderProgram->setUniformValue("gDirectionalLight.Direction", Light.Direction.normalized());
 	m_shaderProgram->setUniformValue("gDirectionalLight.Base.AmbientIntensity", Light.AmbientIntensity);
 	m_shaderProgram->setUniformValue("gDirectionalLight.Base.DiffuseIntensity", Light.DiffuseIntensity);
}


void ShadingTechnique::setMVPMatrix(const mat4& WVP)
{
	m_shaderProgram->setUniformValue("gWVP", WVP);
}

void ShadingTechnique::setWorldMatrix(const mat4& World)
{
	m_shaderProgram->setUniformValue("gWorld", World);
}


void ShadingTechnique::setColorTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gColorMap", TextureUnit);
}

void ShadingTechnique::setShadowMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gShadowMap", TextureUnit);
}

void ShadingTechnique::setNormalMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gNormalMap", TextureUnit);
}

void ShadingTechnique::setEyeWorldPos(const vec3& EyeWorldPos)
{
	m_shaderProgram->setUniformValue("gEyeWorldPos", EyeWorldPos);
}


void ShadingTechnique::setMatSpecularIntensity(float Intensity)
{
	m_shaderProgram->setUniformValue("gMatSpecularIntensity", Intensity);
}


void ShadingTechnique::setMatSpecularPower(float Power)
{
	m_shaderProgram->setUniformValue("gSpecularPower", Power);
}

void ShadingTechnique::setPointLights(unsigned int NumLights, const PointLight* pLights)
{
    glUniform1i(m_numPointLightsLocation, NumLights);
    
    for (unsigned int i = 0 ; i < NumLights ; i++) {
        glUniform3f(m_pointLightsLocation[i].Color, pLights[i].Color.x(), pLights[i].Color.y(), pLights[i].Color.z());
        glUniform1f(m_pointLightsLocation[i].AmbientIntensity, pLights[i].AmbientIntensity);
        glUniform1f(m_pointLightsLocation[i].DiffuseIntensity, pLights[i].DiffuseIntensity);
        glUniform3f(m_pointLightsLocation[i].Position, pLights[i].Position.x(), pLights[i].Position.y(), pLights[i].Position.z());
        glUniform1f(m_pointLightsLocation[i].Atten.Constant, pLights[i].Attenuation.Constant);
        glUniform1f(m_pointLightsLocation[i].Atten.Linear, pLights[i].Attenuation.Linear);
        glUniform1f(m_pointLightsLocation[i].Atten.Exp, pLights[i].Attenuation.Exp);
    }
}

void ShadingTechnique::setSpotLights(unsigned int NumLights, const SpotLight* pLights)
{
    glUniform1i(m_numSpotLightsLocation, NumLights);

    for (unsigned int i = 0 ; i < NumLights ; i++) {
        glUniform3f(m_spotLightsLocation[i].Color, pLights[i].Color.x(), pLights[i].Color.y(), pLights[i].Color.z());
        glUniform1f(m_spotLightsLocation[i].AmbientIntensity, pLights[i].AmbientIntensity);
        glUniform1f(m_spotLightsLocation[i].DiffuseIntensity, pLights[i].DiffuseIntensity);
        glUniform3f(m_spotLightsLocation[i].Position,  pLights[i].Position.x(), pLights[i].Position.y(), pLights[i].Position.z());
        vec3 Direction = pLights[i].Direction;
        Direction.normalize();
        glUniform3f(m_spotLightsLocation[i].Direction, Direction.x(), Direction.y(), Direction.z());
        glUniform1f(m_spotLightsLocation[i].Cutoff, cosf(qDegreesToRadians(pLights[i].Cutoff)));
        glUniform1f(m_spotLightsLocation[i].Atten.Constant, pLights[i].Attenuation.Constant);
        glUniform1f(m_spotLightsLocation[i].Atten.Linear,   pLights[i].Attenuation.Linear);
        glUniform1f(m_spotLightsLocation[i].Atten.Exp,      pLights[i].Attenuation.Exp);
    }
}


void ShadingTechnique::setBoneTransform(uint Index, const mat4& Transform)
{
    assert(Index < MAX_BONES);
    glUniformMatrix4fv(m_boneLocation[Index], 1, GL_FALSE, (const GLfloat*)Transform.data());       
}

void ShadingTechnique::setVertexColor( const QColor& col )
{
	// enable customized color first
	m_shaderProgram->setUniformValue("customizedColor", 1);
	m_shaderProgram->setUniformValue("vColor", col);
}

void ShadingTechnique::initLights()
{
	LightPtr light = m_scene->getLight();
	qDebug() << light->position();
}

void ShadingTechnique::setMaterial( const Material& mat )
{
	m_shaderProgram->setUniformValue("material.Ka", mat.m_ambientColor);
	m_shaderProgram->setUniformValue("material.Kd", mat.m_diffuseColor);
	m_shaderProgram->setUniformValue("material.Ks", mat.m_specularColor);
	m_shaderProgram->setUniformValue("material.Ke", mat.m_emissiveColor);
	m_shaderProgram->setUniformValue("material.shininessStrength", mat.m_shininessStrength);
	m_shaderProgram->setUniformValue("material.shininess", mat.m_shininess);
}
