/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkSerializeDeprecated.h>

#include <Common/Serialize/Util/Xml/hkXmlParser.h>

static const char missingFeatureError[] = 
	"Packfile versioning support is not linked. Versioning packfiles at runtime was deprecated in Havok-7.0.0.\n" \
	"To do so requires linking some deprecated code from Source/Common/Compat/Deprecated\n" \
	"If you are using hkProductFeatures.cxx, ensure you do not define HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700.\n" \
	"Note that by default this pulls in a lot of code and data (mainly previous versions of hkClasses).\n" \
	"Some extra effort is required to strip the unused code and data but it will still cost several hundred Kb.\n" \
	"Alternatively, you can use Tools/PackfileConvert/AsseetCc2 to convert your packfiles the the latest version before loading.\n";

hkResult hkSerializeDeprecated::saveXmlPackfile( const void* object, const hkClass& klass, hkStreamWriter* writer, const hkPackfileWriter::Options& options, hkPackfileWriter::AddObjectListener* userListener, hkSerializeUtil::ErrorDetails* errorOut )
{
	if( errorOut )
	{
		errorOut->raiseError(hkSerializeUtil::ErrorDetails::ERRORID_DEPRECATED_NOT_INITIALIZED, "XML packfile support is not linked. Perhaps you have HK_EXCLUDE_FEATURE_SerializeDeprecatedPre700 in hkProductFeatures");
	}
	return HK_FAILURE;
}

hkBool32 hkSerializeDeprecated::isLoadable(const hkSerializeUtil::FormatDetails& details)
{
	return false;
}

hkResource* hkSerializeDeprecated::loadOldPackfile(hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut )
{
	if( errorOut )
	{
		errorOut->raiseError(hkSerializeUtil::ErrorDetails::ERRORID_DEPRECATED_NOT_INITIALIZED, missingFeatureError );
	}
	return HK_NULL;
}

hkObjectResource* hkSerializeDeprecated::loadOldPackfileOnHeap(hkStreamReader& sr, const hkSerializeUtil::FormatDetails& details, hkSerializeUtil::ErrorDetails* errorOut)
{
	if( errorOut )
	{
		errorOut->raiseError(hkSerializeUtil::ErrorDetails::ERRORID_DEPRECATED_NOT_INITIALIZED, missingFeatureError );
	}
	return HK_NULL;
}

hkResult hkSerializeDeprecated::readXmlPackfileHeader(hkStreamReader* stream, XmlPackfileHeader& out, hkSerializeUtil::ErrorDetails* errorOut)
{
	if( errorOut )
	{
		// If the file is really an xml packfile, we raise the ERRORID_DEPRECATED_NOT_INITIALIZED error.
		hkXmlParser parser;
		hkXmlParser::Node* node = HK_NULL;
		// Parse the first node, it should be "hkpackfile".
		if(parser.nextNode(&node, stream) == HK_SUCCESS)
		{
			if(hkXmlParser::StartElement* startElement = node->asStart())
			{
				if(startElement->name == "hkpackfile")
				{
					errorOut->raiseError(hkSerializeUtil::ErrorDetails::ERRORID_DEPRECATED_NOT_INITIALIZED, missingFeatureError );
				}
			}
			node->removeReference();
		}
	}
	return HK_FAILURE;
}

bool hkSerializeDeprecated::isEnabled() const
{
	return false;
}

static hkReferencedObject* createInstance()
{
	static hkSerializeDeprecated s_instance;
	// Prevent this instance from being deleted
	s_instance.addReference();
	return &s_instance;
}
HK_SINGLETON_CUSTOM_CALL(hkSerializeDeprecated, createInstance);

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
