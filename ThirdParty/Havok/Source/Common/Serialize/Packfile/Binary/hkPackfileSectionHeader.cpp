/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>

namespace
{
	template <typename T>
	int extractAndAdvance( const void* p, int& off )
	{
		const T* tp = reinterpret_cast<const T*>( static_cast<const char*>(p) + off );
		off += sizeof(T);
		return *tp;
	}

	const char* extractAndAdvanceString( const void* p, int& off )
	{
		HK_ASSERT(0x4dd0156a, (hkUlong(p)&3) == 0);
		HK_ASSERT(0x7984746d, (off&3) == 0);
		const char* ret = static_cast<const char*>(p) + off;
		int i = 0; // skip string<null>{1}<pad>{0,3}
		for( ; ret[i] != 0; ++i ) { } // string body
		i += 1; // skip null
		for( ; (i&3) != 0; ++i ) { } // pad up to int32
		off += i;
		return ret;
	}
}

void hkPackfileSectionHeader::getExports( const void* sectionBegin, hkArray<hkResource::Export>& exportsOut ) const
{
	const void* exportsBase = hkAddByteOffsetConst(sectionBegin, m_exportsOffset);
	for( int i = 0; i < getExportsSize(); /**/ )
	{
		int off = extractAndAdvance<hkInt32>(exportsBase, i);
		if( off == -1 )
		{
			break;
		}
		HK_ASSERT(0x4208950c, unsigned(off) < unsigned(getDataSize()) );
		const char* name = extractAndAdvanceString(exportsBase, i);
		hkResource::Export& e = exportsOut.expandOne();
		e.name = name;
		e.data = const_cast<void*>( hkAddByteOffsetConst(sectionBegin, off) );
	}
}

void hkPackfileSectionHeader::getImports( const void* sectionBegin, hkArray<hkResource::Import>& importsOut ) const
{
	const void* imports = hkAddByteOffsetConst(sectionBegin, m_importsOffset);
	for( int i = 0; i < getImportsSize(); /**/ )
	{
		int off = extractAndAdvance<hkInt32>(imports, i);
		if( off == -1 )
		{
			break;
		}
		HK_ASSERT(0x3b2e4f83, unsigned(off) < unsigned(getDataSize()) );
		HK_ASSERT(0xd207ae6b, (off & (sizeof(void*)-1)) == 0 );
		const char* name = extractAndAdvanceString(imports, i);
		hkResource::Import& imp = importsOut.expandOne();
		imp.name = name;
		imp.location = const_cast<void**>( reinterpret_cast<void*const*>( hkAddByteOffsetConst(sectionBegin, off) ) );
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
