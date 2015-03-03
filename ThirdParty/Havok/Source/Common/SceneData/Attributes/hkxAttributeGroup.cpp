/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Attributes/hkxAttributeGroup.h>
#include <Common/Base/Reflection/hkClass.h>

hkxAttributeGroup& hkxAttributeGroup::operator=( const hkxAttributeGroup& other )
{
	m_name = other.m_name;
	m_attributes.setSize(0);
	if (other.m_attributes.getSize())
		m_attributes.insertAt(0, other.m_attributes.begin(), other.m_attributes.getSize());

	return *this;
}

hkResult hkxAttributeGroup::getBoolValue(const char* name, bool warnIfNotFound, hkBool& boolOut) const
{
	// Bool
	{
		hkxSparselyAnimatedBool* data = findBoolAttributeByName(name);
		if (data) 
		{
			boolOut=data->m_bools[0];
			return HK_SUCCESS;
		}
	}

	// Int
	{
		hkxSparselyAnimatedInt* data = findIntAttributeByName(name);
		if (data) 
		{
			boolOut = (data->m_ints[0] != 0);
			return HK_SUCCESS;
		}
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Bool attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getIntValue(const char* name, bool warnIfNotFound, int& intOut) const
{
	// Int
	{
		hkxSparselyAnimatedInt* data = findIntAttributeByName(name);
		if (data) 
		{
			intOut = data->m_ints[0];
			return HK_SUCCESS;
		}
	}

	// Bool
	{
		hkxSparselyAnimatedBool* data = findBoolAttributeByName(name);
		if (data) 
		{
			intOut=data->m_bools[0] ? 1 : 0;
			return HK_SUCCESS;
		}
	}

	// Enum
	{
		hkxSparselyAnimatedEnum* data = findEnumAttributeByName(name);
		if (data) 
		{
			intOut = data->m_ints[0];
			return HK_SUCCESS;
		}
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Integer attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getIntValue (const char* name, bool warnIfNotFound, hkUint32& intOut) const
{
	// We treat the same as ints
	return getIntValue(name, warnIfNotFound, (int&) (intOut));
}

hkResult hkxAttributeGroup::getStringValue(const char* name, bool warnIfNotFound, const char*& stringOut) const
{
	// String
	{
		hkxSparselyAnimatedString* data = findStringAttributeByName(name);
		if (data)
		{
			stringOut = data->m_strings[0].cString();
			return HK_SUCCESS;
		}
	}

	// Enum
	{
		hkxSparselyAnimatedEnum* data = findEnumAttributeByName(name);
		if (data) 
		{
			int intValue = data->m_ints[0];
			data->m_enum->getNameOfValue(intValue, &stringOut);

			return HK_SUCCESS;
		}
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "String attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getFloatValue(const char* name, bool warnIfNotFound, float& floatOut) const
{
	hkxAnimatedFloat* data = findFloatAttributeByName(name);
	if (data) 
	{
		floatOut = data->m_floats[0];
		return HK_SUCCESS;
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Float attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getVectorValue(const char* name, bool warnIfNotFound, hkVector4& vectorOut) const
{
	hkxAnimatedVector* data = findVectorAttributeByName(name);
	if (data) 
	{
		vectorOut.load<4,HK_IO_NATIVE_ALIGNED>(&data->m_vectors[0]);
		return HK_SUCCESS;
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Float attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getQuaternionValue(const char* name, bool warnIfNotFound, hkQuaternion& quaternionOut) const
{
	hkxAnimatedQuaternion* data = findQuaternionAttributeByName(name);
	if (data) 
	{
		quaternionOut.m_vec.load<4,HK_IO_NATIVE_ALIGNED>(&data->m_quaternions[0]);
		return HK_SUCCESS;
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Quaternion attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

hkResult hkxAttributeGroup::getMatrixValue(const char* name, bool warnIfNotFound, hkMatrix4& matrixOut) const
{
	hkxAnimatedMatrix* data = findMatrixAttributeByName(name);
	if (data) 
	{
		matrixOut.set4x4ColumnMajor(&data->m_matrices[0]);
		return HK_SUCCESS;
	}

	if (warnIfNotFound)
	{
		HK_WARN_ALWAYS (0xabbaab81, "Matrix attribute "<<name<<" not found in "<<m_name<<" attribute group");
	}

	return HK_FAILURE;
}

int hkxAttributeGroup::findAttributeIndexByName( const char* name ) const
{
	for (int i=0; i < m_attributes.getSize(); ++i)
	{
		if (hkString::strCasecmp(m_attributes[i].m_name, name) == 0 )
		{
			return i;
		}
	}
	return -1;
}

hkRefVariant hkxAttributeGroup::findAttributeVariantByName(const char* name ) const
{
	const int index = findAttributeIndexByName( name );

	hkVariant v  = {HK_NULL, HK_NULL};

	return (index<0) ? hkRefVariant(v) : m_attributes[index].m_value;
}

hkReferencedObject* hkxAttributeGroup::findAttributeObjectByName(const char* name, const hkClass* type ) const
{
	hkRefVariant var = findAttributeVariantByName(name );

	// compare class by name so that it deals with serialized classes etc better (for instance in the filters)
	if (var && (!type || (hkString::strCasecmp(type->getName(), var.getClass()->getName()) == 0)) )
	{
		return var;
	}
	return HK_NULL;
}

class hkxSparselyAnimatedBool* hkxAttributeGroup::findBoolAttributeByName (const char* name) const
{
	return static_cast<hkxSparselyAnimatedBool*> (findAttributeObjectByName(name, &hkxSparselyAnimatedBoolClass));

}

class hkxSparselyAnimatedInt* hkxAttributeGroup::findIntAttributeByName (const char* name) const
{
	return static_cast<hkxSparselyAnimatedInt*> (findAttributeObjectByName(name, &hkxSparselyAnimatedIntClass));

}

class hkxSparselyAnimatedEnum* hkxAttributeGroup::findEnumAttributeByName (const char* name) const
{
	return static_cast<hkxSparselyAnimatedEnum*> (findAttributeObjectByName(name, &hkxSparselyAnimatedEnumClass));

}

class hkxSparselyAnimatedString* hkxAttributeGroup::findStringAttributeByName (const char* name) const
{
	return static_cast<hkxSparselyAnimatedString*> (findAttributeObjectByName(name, &hkxSparselyAnimatedStringClass));

}

class hkxAnimatedFloat* hkxAttributeGroup::findFloatAttributeByName (const char* name) const
{
	return static_cast<hkxAnimatedFloat*> (findAttributeObjectByName(name, &hkxAnimatedFloatClass));

}

class hkxAnimatedVector* hkxAttributeGroup::findVectorAttributeByName (const char* name) const
{
	return static_cast<hkxAnimatedVector*> (findAttributeObjectByName(name, &hkxAnimatedVectorClass));
}

class hkxAnimatedQuaternion* hkxAttributeGroup::findQuaternionAttributeByName (const char* name) const
{
	return static_cast<hkxAnimatedQuaternion*> (findAttributeObjectByName(name, &hkxAnimatedQuaternionClass));

}

class hkxAnimatedMatrix* hkxAttributeGroup::findMatrixAttributeByName (const char* name) const
{
	return static_cast<hkxAnimatedMatrix*> (findAttributeObjectByName(name, &hkxAnimatedMatrixClass));

}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
