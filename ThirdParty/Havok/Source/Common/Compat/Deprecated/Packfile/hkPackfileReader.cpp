/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/System/Io/Reader/hkStreamReader.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileHeader.h>
#include <Common/Compat/Deprecated/Packfile/hkPackfileReader.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>

#if 0 && defined(HK_DEBUG)
#	include <Common/Base/Fwd/hkcstdio.h>
	using namespace std;
#	define TRACE(A) A
#else
#	define TRACE(A) // nothing
#endif

extern const hkClass hkClassVersion1Class;


hkPackfileReader::hkPackfileReader()
	: m_contentsVersion(HK_NULL)
{
}

hkPackfileReader::~hkPackfileReader()
{
	hkDeallocate(m_contentsVersion);
}

void* hkPackfileReader::getContents( const char* className )
{
	hkBuiltinTypeRegistry& builtin = hkBuiltinTypeRegistry::getInstance();
	return getContentsWithRegistry( className, builtin.getLoadedObjectRegistry() );
}

void hkPackfileReader::updateMetaDataInplace( hkClass* classInOut, int fileVersion, const char* contentsVersion, UpdateFlagFromClassMap& updateFlagFromClass )
{
	hkClass::updateMetadataInplace(classInOut, updateFlagFromClass, fileVersion );
}

const char* hkPackfileReader::getContentsVersion() const
{
	return m_contentsVersion;
}

void hkPackfileReader::setContentsVersion(const char* ver)
{
	hkDeallocate(m_contentsVersion);
	m_contentsVersion = hkString::strDup(ver);
}

hkBool32 hkPackfileReader::isVersionUpToDate() const
{
	// Accept any file newer than 7.0.0-r1
	return hkString::strCmp(m_contentsVersion, hkVersionUtil::getDeprecatedCurrentVersion()) >= 0;
}

void hkPackfileReader::warnIfNotUpToDate() const
{
	if( !isVersionUpToDate() )
	{
		HK_WARN_ALWAYS(0x7aef6c06, "Loaded data contains version " << m_contentsVersion 
			<< " but the current version is " << hkVersionUtil::getDeprecatedCurrentVersion()
			<< ". Did you call hkVersionUtil::updateToCurrentVersion() or did it fail?" );
	}
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
