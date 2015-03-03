/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/Util/hkVersionCheckingUtils.h>
#include <Common/Compat/Deprecated/Version/hkVersionCheckingUtilsOld.h>
#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>
#include <Common/Compat/Deprecated/Compat/hkHavokAllClassUpdates.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Compat/hkHavokVersions.h>

namespace 
{
	struct DeferredErrorStream : public hkStreamWriter
	{
		int write(const void* buf, int nbytes)	{ m_data.insertAt( m_data.getSize(), (char*)buf, nbytes ); return nbytes; }

		hkBool isOk() const { return true; }

		void clear() { m_data.clear(); }

		void dump() 
		{ 
			m_data.pushBack('\0'); // ensure termination
			hkError::getInstance().message(hkError::MESSAGE_REPORT, 1, m_data.begin(), __FILE__, __LINE__);
		}

		hkArray<char> m_data;
	};
}

static int versioningCheck()
{
	DeferredErrorStream deferred;
	hkOstream output(&deferred);
	const hkClassNameRegistry* oldClassReg;
	const hkClassNameRegistry* newClassReg;

	const char* ignoredPrefixes[] = { "hkai" };
	const int numIgnoredPrefixes = HK_COUNT_OF(ignoredPrefixes);

	hkResult res;
	hkError::getInstance().setEnabled(0x786cb087, false);
#	define HK_CLASS_UPDATE_INFO(OLD,NEW) \
	oldClassReg = hkVersionRegistry::getInstance().getClassNameRegistry(hkHavok##OLD##Classes::VersionString); \
	HK_ASSERT(0x16e1254c, oldClassReg); \
	newClassReg = hkVersionRegistry::getInstance().getClassNameRegistry(hkHavok##NEW##Classes::VersionString); \
	HK_ASSERT(0x43014aa0, newClassReg); \
	deferred.clear(); \
	res = hkVersionCheckingUtils::verifyUpdateDescription( output, *oldClassReg, *newClassReg, hkCompat_hk##OLD##_hk##NEW ::hkVersionUpdateDescription, hkVersionCheckingUtils::IGNORE_REMOVED, ignoredPrefixes, numIgnoredPrefixes ); \
	if (res!=HK_SUCCESS) { deferred.dump(); } \
	HK_TEST2( res == HK_SUCCESS, #OLD" -> "#NEW );

#	define HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE HK_HAVOK_VERSION_300
#	include <Common/Compat/Deprecated/Compat/hkCompatVersions.h>
#	undef HK_CLASS_UPDATE_INFO
#	undef HK_SERIALIZE_MIN_COMPATIBLE_VERSION_INTERNAL_VALUE

	hkError::getInstance().setEnabled(0x786cb087, true);
	return 0;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(versioningCheck, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
