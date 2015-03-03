/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKCD_PLANAR_CSG_OPERAND_H
#define HKCD_PLANAR_CSG_OPERAND_H

#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>
#include <Geometry/Collide/DataStructures/Planar/Memory/hkcdPlanarGeometryPrimitivesCollectionManager.h>

/// CSG Operand
class hkcdPlanarCsgOperand : public hkReferencedObject
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);

		// Types
		typedef hkcdPlanarGeometryPlanesCollection	PlanesCollection;
		typedef hkcdPlanarGeometryPrimitives::CollectionManager<hkcdPlanarGeometryPolygonCollection>	PolyCollManager;
		typedef hkcdPlanarGeometryPrimitives::CollectionManager<hkcdConvexCellsCollection>				CellCollManager;
		typedef hkcdPlanarGeometryPrimitives::CollectionManager<hkcdPlanarSolid::ArrayMgr>				ArrayCollManager;

		/// Holder for geometry extra infos
		struct GeomExtraInfos : public hkReferencedObject
		{
			HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);
			hkArray<hkcdConvexCellsTree3D::PolygonId> m_danglingPolyIds;
		};

		/// Holder for geometry source
		struct GeomSource : public hkReferencedObject
		{
			public:

				HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);

			public:

				/// Constructor
				GeomSource();

				/// Constructor
				GeomSource(const hkcdPlanarCsgOperand& operand, const int nbMatIds);

			public:

				hkRefPtr<hkcdPlanarGeometry> m_geometry;	///< Source geometry
				int m_materialOffset;						///< Offset for material remapping
				int m_numMaterialIds;						///< For material remapping
				hkRefPtr<GeomExtraInfos> m_geomInfos;		///< Dangling polygons
				hkRefPtr<hkcdPlanarSolid> m_cutoutSolid;	///< Cutout to be applied on the dangling polygons (if any...)
				bool m_flipPolygons;						///< True if the flipped version of the geometry should be used
		};

	public:

		/// Constructor
		hkcdPlanarCsgOperand();

		/// Destructor
		virtual ~hkcdPlanarCsgOperand();

	public:

		/// Returns the convex cell tree corresponding to this solid planar geom. Build it if necessary.
		hkcdConvexCellsTree3D* getOrCreateConvexCellTree(bool withConnectivity = false, bool rebuildIfConnectivityDoesntMatch = true);

		/// Retrieves the planes collection
		HK_FORCE_INLINE PlanesCollection* accessPlanesCollection();
		const PlanesCollection* getPlanesCollection() const;

		/// Sets a new planes collection. If the plane remapping table is non-null, the plane Ids on all nodes will be re-set as well (i.e. to match the plane Ids in the new collection)
		void setPlanesCollection(const PlanesCollection* newPlanes, const int* HK_RESTRICT planeRemapTable);

		/// Shift all plane ids of the operand elements
		void shiftPlaneIds(int offsetValue);

		/// Simplifies this operand by rebuilding the solid BSP tree from its boundaries
		void simplifyFromBoundaries();

		/// Build the planar geometry from its geometry sources
		void buildGeometryFromGeomSources(bool useStandardClassify, bool intersectCoplanarPolygons);

		/// Fast version of buildGeometryFromGeomSources when only 2 geometry sources are present
		void buildGeometryFrom2GeomSources(PolyCollManager* polysCollManager, ArrayCollManager* arraysCollManager);

		/// Create a unique temporary instance of the ith geom source matching the operand planes collection
		void createGeometryFromGeometrySource(int geomSourceId, hkcdPlanarGeometry* geomDest, hkUint32 maxPlaneIdValue, hkArray<hkcdConvexCellsTree3D::PolygonId>& polyIds, hkArray<hkcdConvexCellsTree3D::PolygonId>& danglingPolyIds, hkArray<int>& remapTable);

		/// Removes all planes not used by the entities
		void removeUnusedPlanes();

		/// Copy the desired data from another operand
		void copyData(const hkcdPlanarCsgOperand& operandSrc, bool copySolid, bool copyRegions, bool copyGeometry = false);

		/// Copy the desired data from another operand, using provided collection managers
		void copyData(	const hkcdPlanarCsgOperand& operandSrc, hkcdPlanarGeometryPlanesCollection* dstPlaneCollection,
						PolyCollManager* polysCollManager, CellCollManager* cellsCollManager, bool copyRegions = true, bool copyGeometry = false);

		/// Create an operand with the same data, but a duplicated plane collection
		static void HK_CALL createOperandWithSharedDataAndClonedPlanes(const hkcdPlanarCsgOperand* operandSrc, hkRefPtr<hkcdPlanarCsgOperand>& operandDst);

	protected:

		/// Collects a bit-field of plane Ids used by the operand
		void collectUsedPlaneIds(hkBitField& usedPlaneIdsOut) const;

	public:

		/// Gets / sets the geometry
		HK_FORCE_INLINE const hkcdPlanarGeometry* getGeometry() const;
		HK_FORCE_INLINE hkcdPlanarGeometry* accessGeometry();
		HK_FORCE_INLINE void setGeometry(hkcdPlanarGeometry* geom);

		/// Gets / sets the geometry info
		HK_FORCE_INLINE const GeomExtraInfos* getGeomInfos() const;
		HK_FORCE_INLINE GeomExtraInfos* accessGeomInfos();
		HK_FORCE_INLINE void setGeomInfos(GeomExtraInfos* geomInfo);

		/// Gets / sets the solid
		HK_FORCE_INLINE const hkcdPlanarSolid* getSolid() const;
		HK_FORCE_INLINE hkcdPlanarSolid* accessSolid();
		HK_FORCE_INLINE void setSolid(hkcdPlanarSolid* solid);

		/// Gets / sets the regions
		HK_FORCE_INLINE const hkcdConvexCellsTree3D* getRegions() const;
		HK_FORCE_INLINE hkcdConvexCellsTree3D* accessRegions();
		HK_FORCE_INLINE void setRegions(hkcdConvexCellsTree3D* regions);

		/// Gets / sets the geometry sources
		GeomSource* addGeometrySource(const GeomSource& geomSource);
		HK_FORCE_INLINE void appendGeometrySources(const hkArray<GeomSource>& geomSources, bool flipGeoms);
		HK_FORCE_INLINE const hkArray<GeomSource>& getGeometrySources() const;
		HK_FORCE_INLINE hkArray<GeomSource>& accessGeometrySources();
		HK_FORCE_INLINE void removeGeometrySources();

	protected:

		hkRefPtr<hkcdPlanarGeometry> m_geometry;			///< Geometry associated with the fracture piece
		hkRefPtr<GeomExtraInfos> m_geomInfos;				///< Extra information on the held geometry
		hkRefPtr<hkcdPlanarSolid> m_solid;					///< Solid boundary representation of the fracture piece
		hkRefPtr<hkcdConvexCellsTree3D> m_regions;			///< A tree of convex regions matching the solid
		hkArray<GeomSource> m_geomSources;					///< List of geometry sources
};

#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarCsgOperand.inl>

#endif	// HKCD_PLANAR_CSG_OPERAND_H

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
